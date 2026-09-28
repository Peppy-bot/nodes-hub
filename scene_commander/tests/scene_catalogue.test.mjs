import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

// Execute the page served by the Python node. This harness supplies only the
// DOM and HTTP boundary; selection, rendering and requests use the real script.
const source = readFileSync(new URL('../src/scene_commander/__main__.py', import.meta.url), 'utf8');
const html = source.match(/HTML = r"""([\s\S]*?)"""/)[1];
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

class Option {
    constructor(text, value) {
        this.textContent = String(text);
        this.value = String(value);
        this.disabled = false;
    }
}

class ClassList {
    constructor(names) {
        this.names = new Set(names);
    }

    add(name) { this.names.add(name); }
    remove(name) { this.names.delete(name); }
    contains(name) { return this.names.has(name); }

    toggle(name, force) {
        const on = force === undefined ? !this.names.has(name) : force;
        on ? this.names.add(name) : this.names.delete(name);
        return on;
    }
}

class Element {
    constructor(tag, attributes, elements) {
        this.tag = tag;
        this.elements = elements;
        this.generatedIds = [];
        this.attributes = attributes;
        this.value = attributes.value || '';
        this.textContent = '';
        this.hidden = Object.hasOwn(attributes, 'hidden');
        this.disabled = Object.hasOwn(attributes, 'disabled');
        this.classList = new ClassList((attributes.class || '').split(/\s+/).filter(Boolean));
        this.style = {};
        this.children = [];
    }

    replaceChildren(...children) {
        this.children = children;
        if (this.tag === 'select') this.value = children[0]?.value || '';
    }

    get innerHTML() {
        return this.markup || '';
    }

    set innerHTML(value) {
        this.markup = value;
        for (const id of this.generatedIds) this.elements.delete(id);
        const children = registerElements(value, this.elements);
        this.generatedIds = children.map(child => child.attributes.id);
        this.replaceChildren(...children);
    }
}

function registerElements(markup, elements) {
    const children = [];
    for (const [, tag, raw] of markup.matchAll(/<([a-z]+)\b([^>]*\bid="[^>]+)>/g)) {
        const attributes = Object.fromEntries(Array.from(raw.matchAll(/([\w-]+)(?:="([^"]*)")?/g),
            ([, name, value]) => [name, value || '']));
        const element = new Element(tag, attributes, elements);
        elements.set(attributes.id, element);
        children.push(element);
    }
    return children;
}

const staticScene = {
    kind: 'scene', asset_id: 'scene/warehouse_static',
    display_name: 'Warehouse - Static (Fast)',
    description: 'Fixed warehouse objects for faster simulation.',
};
const dynamicScene = {
    kind: 'scene', asset_id: 'scene/warehouse_dynamic',
    display_name: 'Warehouse - Dynamic (Interactive)',
    description: 'Individual movable assets with mass, inertia and rolling cages.',
};
const cage = {
    kind: 'object', asset_id: 'object/warehouse_cage', display_name: 'Rolling Cage',
    category: 'Warehouse', default_mass: 40,
    description: 'Wire transport cage with four jointed wheels for pushing and rolling.',
};
const carton = {
    kind: 'object', asset_id: 'object/warehouse_carton', display_name: 'Cardboard Box',
    category: 'Warehouse', default_mass: 2,
    description: 'Closed corrugated carton for grasping, carrying and stacking.',
};
const block = {
    kind: 'object', asset_id: 'object/red_block', display_name: 'Red Block',
    category: 'Basic', default_mass: 0.1,
    description: 'Small red cube for grasping and stacking.',
};

// The body of a streamed answer, fed by the test: each chunk it pushes is one
// read of the page's reader. It holds the connection its request came on
// until it ends, or until the page aborts that request: as in a browser, the
// abort frees the connection at once and fails every read after it.
class Stream {
    constructor() {
        this.chunks = [];
        this.ended = false;
        this.aborted = false;
        this.abortReason = null;
        this.pendingRead = null;
        this.onRead = null;
        this.release = () => {};
    }

    push(...lines) {
        for (const line of lines) this.pushText(`${JSON.stringify(line)}\n`);
    }

    pushText(text) {
        const chunk = new TextEncoder().encode(text);
        if (!this.pendingRead) {
            this.chunks.push(chunk);
            return;
        }
        this.takePendingRead().resolve({ done: false, value: chunk });
    }

    end() {
        this.ended = true;
        this.freeConnection();
        if (!this.pendingRead) return;
        this.takePendingRead().resolve({ done: true, value: undefined });
    }

    // Hold the connection `release` frees until the stream ends or `signal`
    // aborts the page's request.
    hold(signal, release) {
        this.release = release;
        if (this.ended) this.freeConnection();
        signal?.addEventListener('abort', () => this.abort(signal.reason));
    }

    abort(reason) {
        this.aborted = true;
        this.abortReason = reason;
        this.freeConnection();
        if (!this.pendingRead) return;
        this.takePendingRead().reject(reason);
    }

    freeConnection() {
        const release = this.release;
        this.release = () => {};
        release();
    }

    takePendingRead() {
        const read = this.pendingRead;
        this.pendingRead = null;
        return read;
    }

    getReader() {
        return {
            read: () => {
                if (this.aborted) return Promise.reject(this.abortReason);
                if (this.chunks.length) return Promise.resolve({ done: false, value: this.chunks.shift() });
                if (this.ended) return Promise.resolve({ done: true, value: undefined });
                const read = new Promise((resolve, reject) => { this.pendingRead = { resolve, reject }; });
                this.onRead?.();
                return read;
            },
        };
    }

    // Resolves once the page has read everything pushed and waits for more.
    drained() {
        if (this.pendingRead) return Promise.resolve();
        return new Promise(resolve => {
            this.onRead = () => {
                this.onRead = null;
                resolve();
            };
        });
    }
}

// A streamed answer that has already ended with these lines.
function streamOf(...lines) {
    const stream = new Stream();
    stream.push(...lines);
    stream.end();
    return stream;
}

const PROVIDER_BUSY = 'Isaac is still discovering its asset catalogue';
const NO_OBJECT_STATE = 'Isaac has not loaded its stage yet';
const NO_SCENE = 'no scene is loaded';
const CAPTURED = 1757944800.25;

// unavailable: how many catalogue requests the provider refuses first, as the
// node answers while the engine has no catalogue yet (HTTP 503).
// objectState: the provider's reason while it has no object state, which the
// node answers with HTTP 503; null once it has a snapshot.
// capabilities: what /api/capabilities reports bound beside the scene.
// lighting, materials: what their providers list; null while no scene is
// loaded, which the node answers with HTTP 503. cameras: what /api/cameras
// lists, every camera the simulation renders.
// A load or a spawn answers with a stream that ends at once with success,
// unless the test prepared the answer of the next request to its path.
// Requests share the connections a browser keeps to one server over HTTP/1.1,
// six at most: a request waits for a free one, and a streamed answer holds its
// connection until it ends or the page aborts it.
async function page(catalogue = [staticScene, dynamicScene, cage], objects = [],
    { unavailable = 0, objectState = null, robots = [], capabilities = {}, lighting = null, materials = null, cameras = [] } = {}) {
    const elements = new Map();
    registerElements(html, elements);
    const requests = [];
    const waits = [];
    const intervals = [];
    const prepared = new Map();
    const prepare = (path, answer) => {
        if (!prepared.has(path)) prepared.set(path, []);
        prepared.get(path).push(answer);
    };
    let currentCatalogue = catalogue;
    let currentObjects = objects;
    let currentLighting = lighting;
    let currentMaterials = materials;
    let currentCameras = cameras;
    let objectStateReason = objectState;
    let refusals = unavailable;
    const snapshot = () => ({
        status: elements.get('status').textContent,
        loading: elements.get('status').classList.contains('loading'),
        assetCount: elements.get('assetCount').textContent,
        selects: Object.fromEntries(['sceneSelect', 'categorySelect', 'assetSelect'].map(id => {
            const select = elements.get(id);
            return [id, {
                disabled: select.disabled,
                options: select.children.map(o => ({ text: o.textContent, disabled: o.disabled })),
            }];
        })),
    });
    const refused = (status, message) => ({ ok: false, status, json: async () => ({ success: false, message }) });
    // The browser's connections to the node; a freed one goes to the first
    // request that waits for one.
    const connections = { free: 6, waiting: [] };
    const release = () => {
        const next = connections.waiting.shift();
        if (next) next();
        else connections.free += 1;
    };
    // What the node answers a request to `path`.
    const answerTo = path => {
        const preparedAnswer = prepared.get(path)?.shift();
        if (preparedAnswer) {
            return preparedAnswer;
        }
        if (path === '/api/scene/load' || path === '/api/objects/spawn') {
            return { ok: true, status: 200, body: streamOf({ goal: '1' }, { success: true, message: 'Scene loaded' }) };
        }
        if (path === '/api/assets' && refusals > 0) {
            refusals -= 1;
            return refused(503, PROVIDER_BUSY);
        }
        if (path === '/api/objects' && objectStateReason !== null) {
            return refused(503, objectStateReason);
        }
        if (path === '/api/lighting' && currentLighting === null) {
            return refused(503, NO_SCENE);
        }
        if (path === '/api/materials' && currentMaterials === null) {
            return refused(503, NO_SCENE);
        }
        const data = path === '/api/assets' ? { assets: currentCatalogue }
            : path === '/api/objects'
                ? { objects: currentObjects, count: currentObjects.length, timestamp: CAPTURED }
            : path === '/api/robots' ? { robots, count: robots.length }
            : path === '/api/capabilities'
                ? { lighting: false, materials: false, cameras: false, ...capabilities }
            : path === '/api/lighting' ? { lighting: currentLighting }
            : path === '/api/materials' ? { materials: currentMaterials }
            : path === '/api/cameras'
                ? { message: `${currentCameras.length} cameras rendered`, cameras: currentCameras, count: currentCameras.length }
            : { message: 'Scene loaded' };
        return { ok: true, json: async () => ({ success: true, ...data }) };
    };
    const context = vm.createContext({
        AbortController,
        Option,
        TextDecoder,
        document: {
            getElementById: id => {
                assert.ok(elements.has(id), `page contains #${id}`);
                return elements.get(id);
            },
            visibilityState: 'visible',
        },
        setTimeout: (callback, delay) => {
            waits.push({ delay, ...snapshot() });
            callback();
        },
        // A periodic refresh is recorded, never fired: every read under test
        // is one the page makes on its own or one a test asks for.
        setInterval: (callback, delay) => {
            intervals.push(delay);
        },
        // A request reaches the node, and is recorded, once it has a connection.
        fetch: async (path, options) => {
            if (connections.free > 0) {
                connections.free -= 1;
            }
            else {
                await new Promise(resolve => connections.waiting.push(resolve));
            }
            requests.push({ path, body: options.body && JSON.parse(options.body) });
            const answered = await answerTo(path);
            if (answered.body instanceof Stream) {
                answered.body.hold(options.signal, release);
            }
            else {
                release();
            }
            return answered;
        },
    });
    await vm.runInContext(script, context);
    const element = id => elements.get(id);
    return {
        element, requests, waits, intervals,
        run: expression => vm.runInContext(expression, context),
        selectScene: async id => {
            element('sceneSelect').value = id;
            await vm.runInContext(element('sceneSelect').attributes.onchange, context);
        },
        selectAsset: async id => {
            element('assetSelect').value = id;
            await vm.runInContext(element('assetSelect').attributes.onchange, context);
        },
        setCatalogue: assets => { currentCatalogue = assets; },
        setObjects: list => { currentObjects = list; objectStateReason = null; },
        setObjectStateUnavailable: reason => { objectStateReason = reason; },
        setLighting: state => { currentLighting = state; },
        setMaterials: state => { currentMaterials = state; },
        setCameras: list => { currentCameras = list; },
        // The stream the next request to `path` answers with, which the test feeds.
        stream: path => {
            const stream = new Stream();
            prepare(path, { ok: true, status: 200, body: stream });
            return stream;
        },
        // The JSON answer of the next request to `path`, a success.
        answer: (path, data) => prepare(path, { ok: true, status: 200, json: async () => ({ success: true, ...data }) }),
        // The next request to `path` is refused before any stream, as the
        // node refuses bad input and a goal the simulator rejects.
        refuse: (path, status, message) => prepare(path, refused(status, message)),
        // The next request to `path` waits until the test refuses it with
        // the function returned.
        refuseLater: path => {
            let answer;
            prepare(path, new Promise(resolve => { answer = resolve; }));
            return (status, message) => answer(refused(status, message));
        },
        click: id => vm.runInContext(element(id).attributes.onclick, context),
    };
}

test('fetched scenes expose their catalogue names and selected descriptions', async () => {
    const ui = await page();
    assert.deepEqual(ui.requests.map(r => r.path), ['/api/capabilities', '/api/assets', '/api/objects', '/api/robots']);
    assert.deepEqual(Array.from(ui.element('sceneSelect').children, o => o.textContent),
        [staticScene.display_name, dynamicScene.display_name]);
    assert.equal(ui.element('sceneDescription').textContent, staticScene.description);
    assert.equal(ui.element('sceneDescription').hidden, false);
    assert.equal(ui.element('sceneSelect').attributes['aria-describedby'], 'sceneDescription');

    await ui.selectScene(dynamicScene.asset_id);
    assert.equal(ui.element('sceneDescription').textContent, dynamicScene.description);
    assert.equal(ui.element('sceneDescription').hidden, false);
    ui.element('sceneScale').value = '1.5';
    await ui.run('loadScene()');
    assert.deepEqual(ui.requests.at(-2), {
        path: '/api/scene/load', body: { asset_id: dynamicScene.asset_id, scale: 1.5 },
    });
});

test('the page opens in its loading state before any script runs', () => {
    const elements = new Map();
    registerElements(html, elements);
    assert.equal(elements.get('status').classList.contains('loading'), true);
    for (const id of ['sceneSelect', 'categorySelect', 'assetSelect']) {
        assert.equal(elements.get(id).disabled, true, `#${id} starts disabled`);
    }
    assert.match(html, /<select id="sceneSelect"[^>]*>\s*<option value="" disabled>Loading assets\.\.\.<\/option>/);
});

test('the page keeps asking the provider for its catalogue and shows why it waits', async () => {
    const ui = await page([staticScene, cage], [], { unavailable: 2 });
    assert.deepEqual(ui.requests.map(r => r.path),
        ['/api/capabilities', '/api/assets', '/api/assets', '/api/assets', '/api/objects', '/api/robots']);
    assert.equal(ui.waits.length, 2);
    for (const wait of ui.waits) {
        assert.equal(wait.delay, 3000);
        assert.equal(wait.status, `Waiting for the scene provider: ${PROVIDER_BUSY}`);
        assert.equal(wait.loading, true);
        assert.equal(wait.assetCount, 'Loading assets...');
        for (const [id, select] of Object.entries(wait.selects)) {
            assert.equal(select.disabled, true, `#${id} disabled while waiting`);
            assert.deepEqual(select.options, [{ text: 'Loading assets...', disabled: true }]);
        }
    }
    assert.equal(ui.element('status').textContent, 'Loaded 2 assets');
    assert.equal(ui.element('status').classList.contains('loading'), false);
    for (const id of ['sceneSelect', 'categorySelect', 'assetSelect']) {
        assert.equal(ui.element(id).disabled, false, `#${id} enabled once loaded`);
    }
    assert.equal(ui.element('sceneSelect').value, staticScene.asset_id);
    assert.equal(ui.element('assetSelect').value, cage.asset_id);
    assert.equal(ui.element('assetCount').textContent, '1 matching assets');
});

test('an available catalogue is loaded without waiting', async () => {
    const ui = await page();
    assert.deepEqual(ui.waits, []);
    assert.equal(ui.element('status').classList.contains('loading'), false);
    assert.equal(ui.element('status').textContent, 'Loaded 3 assets');
});

test('loading a scene refreshes the runtime objects the provider removed', async () => {
    const ui = await page();
    await ui.run('loadScene()');
    assert.deepEqual(ui.requests.slice(-2).map(r => r.path), ['/api/scene/load', '/api/objects']);
    assert.equal(ui.element('status').textContent, 'Scene loaded');
});

test('refresh retains the chosen scene and replaces its description from the provider', async () => {
    const ui = await page();
    await ui.selectScene(dynamicScene.asset_id);
    ui.setCatalogue([{ ...dynamicScene, description: 'Updated by the scene provider.' }, staticScene]);
    await ui.run('refreshAssets()');
    assert.equal(ui.element('sceneSelect').value, dynamicScene.asset_id);
    assert.equal(ui.element('sceneDescription').textContent, 'Updated by the scene provider.');

    ui.setCatalogue([staticScene]);
    await ui.run('refreshAssets()');
    assert.equal(ui.element('sceneSelect').value, staticScene.asset_id);
    assert.equal(ui.element('sceneDescription').textContent, staticScene.description);
});

test('descriptions are optional and disappear when the provider has no scenes', async () => {
    const ui = await page();
    for (const description of [undefined, '', null]) {
        ui.setCatalogue([{ ...staticScene, description }]);
        await ui.run('refreshAssets()');
        assert.equal(ui.element('sceneDescription').textContent, '');
        assert.equal(ui.element('sceneDescription').hidden, true);
    }
    ui.setCatalogue([cage]);
    await ui.run('refreshAssets()');
    assert.equal(ui.element('sceneSelect').value, '');
    assert.equal(ui.element('sceneDescription').textContent, '');
    assert.equal(ui.element('sceneDescription').hidden, true);
});

test('catalogue names and descriptions remain literal text', async () => {
    const markup = '<img src=x onerror="globalThis.injected = true">';
    const ui = await page([{ ...staticScene, display_name: markup, description: markup }]);
    assert.equal(ui.element('sceneSelect').children[0].textContent, markup);
    assert.equal(ui.element('sceneDescription').textContent, markup);
    assert.deepEqual(ui.element('sceneDescription').children, []);
    assert.equal(ui.run('globalThis.injected'), undefined);
});

test('scene selection keeps catalogue object mass defaults and user mass edits', async () => {
    const ui = await page();
    assert.equal(Number(ui.element('spawnMass').value), cage.default_mass);
    assert.equal(ui.element('spawnPhysics').value, 'dynamic');
    ui.element('spawnMass').value = '55';
    await ui.selectScene(dynamicScene.asset_id);
    await ui.run('refreshAssets()');
    assert.equal(ui.element('assetSelect').value, cage.asset_id);
    assert.equal(Number(ui.element('spawnMass').value), 55);
});


test('asset selection exposes descriptions and applies each selected mass default', async () => {
    const ui = await page([cage, carton]);
    assert.equal(ui.element('assetDescription').textContent, cage.description);
    assert.equal(ui.element('assetDescription').hidden, false);
    assert.equal(ui.element('assetSelect').attributes['aria-describedby'], 'assetDescription');

    await ui.selectAsset(carton.asset_id);
    assert.equal(ui.element('assetDescription').textContent, carton.description);
    assert.equal(Number(ui.element('spawnMass').value), carton.default_mass);
    assert.equal(ui.element('spawnPhysics').value, 'dynamic');
});

test('catalogue refresh updates descriptions while preserving selection, filters and spawn edits', async () => {
    const ui = await page([cage, carton, block]);
    await ui.selectAsset(carton.asset_id);
    ui.element('categorySelect').value = 'Warehouse';
    ui.element('assetSearch').value = 'stacking';
    ui.element('spawnMass').value = '7';
    ui.element('spawnPhysics').value = 'static';
    ui.setCatalogue([cage, { ...carton, description: 'Updated carton for stacking.', default_mass: 3 }, block]);
    await ui.run('refreshAssets()');

    assert.equal(ui.element('assetSelect').value, carton.asset_id);
    assert.equal(ui.element('categorySelect').value, 'Warehouse');
    assert.equal(ui.element('assetSearch').value, 'stacking');
    assert.equal(ui.element('assetDescription').textContent, 'Updated carton for stacking.');
    assert.equal(Number(ui.element('spawnMass').value), 7);
    assert.equal(ui.element('spawnPhysics').value, 'static');
});

test('description search and category filters update the visible description or hide it for no matches', async () => {
    const ui = await page([cage, carton, block]);
    ui.element('assetSearch').value = 'STACKING';
    await ui.run('renderAssets()');
    assert.deepEqual(Array.from(ui.element('assetSelect').children, o => o.value),
        [carton.asset_id, block.asset_id]);
    assert.equal(ui.element('assetDescription').textContent, carton.description);

    ui.element('categorySelect').value = 'Basic';
    await ui.run('renderAssets()');
    assert.equal(ui.element('assetSelect').value, block.asset_id);
    assert.equal(ui.element('assetDescription').textContent, block.description);

    ui.element('assetSearch').value = 'no matching description';
    await ui.run('renderAssets()');
    assert.equal(ui.element('assetSelect').value, '');
    assert.equal(ui.element('assetDescription').textContent, '');
    assert.equal(ui.element('assetDescription').hidden, true);
});

test('asset descriptions are optional and clear when assets disappear', async () => {
    const ui = await page([cage]);
    for (const description of [undefined, '', null]) {
        ui.setCatalogue([{ ...cage, description }]);
        await ui.run('refreshAssets()');
        assert.equal(ui.element('assetDescription').textContent, '');
        assert.equal(ui.element('assetDescription').hidden, true);
    }
    ui.setCatalogue([staticScene]);
    await ui.run('refreshAssets()');
    assert.equal(ui.element('assetSelect').value, '');
    assert.equal(ui.element('assetDescription').textContent, '');
    assert.equal(ui.element('assetDescription').hidden, true);
});

test('asset names, categories and descriptions remain literal catalogue text', async () => {
    const markup = '<img src=x onerror="globalThis.injected = true">';
    const ui = await page([{ ...cage, display_name: markup, category: markup, description: markup }]);
    assert.equal(ui.element('assetSelect').children[0].textContent, `${markup} - ${markup}`);
    assert.equal(ui.element('categorySelect').children[1].textContent, markup);
    assert.equal(ui.element('assetDescription').textContent, markup);
    assert.deepEqual(ui.element('assetDescription').children, []);
    assert.equal(ui.run('globalThis.injected'), undefined);
});

test('runtime objects show their asset descriptions and refresh them without losing position edits', async () => {
    const objects = [
        { object_id: 'cage_1', asset_id: cage.asset_id, physics: 'dynamic', mass: 40, position: [1, 2, 0] },
        { object_id: 'unknown_1', asset_id: 'object/unknown', physics: 'static' },
    ];
    const ui = await page([cage], objects);
    assert.equal(ui.element('objectDescription0').textContent, cage.description);
    assert.equal(ui.element('objectDescription0').hidden, false);
    assert.equal(ui.element('objectDescription1').textContent, '');
    assert.equal(ui.element('objectDescription1').hidden, true);

    ui.element('ox0').value = '9';
    ui.element('forceMag0').value = '12';
    const markup = '<img src=x onerror="globalThis.injected = true">';
    ui.setCatalogue([{ ...cage, description: markup }]);
    await ui.run('refreshAssets()');
    assert.equal(ui.element('objectDescription0').textContent, markup);
    assert.deepEqual(ui.element('objectDescription0').children, []);
    assert.equal(ui.run('globalThis.injected'), undefined);
    assert.equal(ui.element('ox0').value, '9');
    assert.equal(ui.element('forceMag0').value, '12');

    ui.setCatalogue([]);
    await ui.run('refreshAssets()');
    assert.equal(ui.element('objectDescription0').textContent, '');
    assert.equal(ui.element('objectDescription0').hidden, true);
});

test('the robot picker lists what the scene stands and a move carries the one chosen', async () => {
    const robots = [
        { robot: 'alpha', model: 'openarm_v2', position: [0, 0, 0], yaw: 0, attached: true },
        { robot: 'bravo', model: 'openarm_v1', position: [0, -1.5, 0], yaw: 1.25, attached: true },
    ];
    const ui = await page([cage], [], { robots });

    const select = ui.element('robotSelect');
    assert.deepEqual(select.children.map(o => o.textContent), [
        'alpha (openarm_v2)',
        'bravo (openarm_v1)',
    ]);
    assert.equal(select.disabled, false);
    assert.equal(ui.element('robotCount').textContent, '2 robots standing');

    // Choosing a robot offers that robot's own position and yaw, not the
    // last one's.
    select.value = 'bravo';
    await ui.run('selectRobot()');
    assert.deepEqual(
        ['robotX', 'robotY', 'robotZ', 'robotYaw'].map(id => String(ui.element(id).value)),
        ['0', '-1.5', '0', '1.25'],
    );

    // A move that only translates sends the yaw the robot is listed at, so
    // the robot keeps facing as it did.
    ui.element('robotX').value = '1';
    await ui.run('moveRobot()');
    const moves = () => ui.requests.filter(r => r.path === '/api/robot/move');
    assert.deepEqual(moves().at(-1).body, { robot: 'bravo', position: [1, -1.5, 0], yaw: 1.25 });

    // A yaw the form names is the one sent. The move before re-read the
    // listing, so the boxes hold where the scene lists the robot again.
    ui.element('robotYaw').value = '-0.5';
    await ui.run('moveRobot()');
    assert.deepEqual(moves().at(-1).body, { robot: 'bravo', position: [0, -1.5, 0], yaw: -0.5 });
});

test('a move waits for a robot to be chosen', async () => {
    const ui = await page([cage], [], { robots: [] });

    assert.equal(ui.element('robotSelect').disabled, true);
    await ui.run('moveRobot()');
    assert.equal(ui.requests.some(r => r.path === '/api/robot/move'), false);
    assert.equal(ui.element('status').textContent, 'select a robot to move');
});

const redBlock = {
    object_id: 'obj_1', asset_id: block.asset_id, physics: 'dynamic', mass: 0.2, scale: 1.5,
    position: [0.5, 0, 0.8], orientation: [0, 0, 0, 1], linear_velocity: [0, 0, 0], angular_velocity: [0, 0, 0],
};
const staticCarton = {
    ...redBlock, object_id: 'obj_2', asset_id: carton.asset_id, physics: 'static', mass: 2, scale: 1,
};

// The ids the objects panel holds right now; the page source names every id
// its templates can render, so the element map alone cannot tell.
const panel = ui => ui.element('objects').children.map(child => child.attributes.id);

test('an empty scene lists no runtime objects', async () => {
    const ui = await page([block], []);
    assert.equal(ui.element('objectCount').textContent, '0 objects');
    assert.match(ui.element('objects').innerHTML, /No runtime objects\./);
    assert.deepEqual(panel(ui), []);
});

test('runtime objects show their physics, mass and scale as spawned', async () => {
    const ui = await page([block, carton], [redBlock, staticCarton]);
    assert.equal(ui.element('objectCount').textContent, '2 objects');
    const markup = ui.element('objects').innerHTML;
    assert.match(markup, /physics=dynamic\s*&nbsp;\|&nbsp;\s*mass=0\.2 kg\s*&nbsp;\|&nbsp;\s*scale=1\.5/);
    assert.match(markup, /physics=static\s*&nbsp;\|&nbsp;\s*mass=2 kg\s*&nbsp;\|&nbsp;\s*scale=1\b/);
    assert.equal(ui.element('ox0').value, '0.5');
    assert.ok(ui.element('forceMag0'), 'a dynamic object keeps its force controls');
    assert.equal(ui.element('forceMag1'), undefined, 'a static object has no force controls');
});

test('unavailable object state is shown as such, never as an empty scene', async () => {
    const ui = await page([block], [], { objectState: NO_OBJECT_STATE });
    assert.deepEqual(ui.requests.map(r => r.path), ['/api/capabilities', '/api/assets', '/api/objects', '/api/robots']);
    assert.equal(ui.element('objectCount').textContent, 'unavailable');
    assert.deepEqual(panel(ui), ['objectsUnavailable']);
    assert.equal(ui.element('objectsUnavailable').textContent, `Object state unavailable: ${NO_OBJECT_STATE}`);
    assert.doesNotMatch(ui.element('objects').innerHTML, /No runtime objects/);
    assert.equal(ui.element('status').textContent, `Object state unavailable: ${NO_OBJECT_STATE}`);
    assert.equal(ui.element('status').style.borderColor, '#9b424c');

    ui.setObjects([]);
    await ui.run('refreshObjects()');
    assert.equal(ui.element('objectCount').textContent, '0 objects');
    assert.match(ui.element('objects').innerHTML, /No runtime objects\./);
    assert.deepEqual(panel(ui), []);
});

test('object state that becomes unavailable drops the controls of the objects it listed', async () => {
    const ui = await page([block], [redBlock]);
    assert.ok(ui.element('ox0'));

    const markup = '<img src=x onerror="globalThis.injected = true">';
    ui.setObjectStateUnavailable(markup);
    await ui.run('refreshObjects()');
    assert.equal(ui.run('objectList.length'), 0);
    for (const id of ['ox0', 'oy0', 'oz0', 'forceMag0', 'forceDuration0', 'objectDescription0']) {
        assert.equal(ui.element(id), undefined, `#${id} is gone with the state it showed`);
    }
    assert.doesNotMatch(ui.element('objects').innerHTML, /Move|Remove|obj_1/);
    assert.equal(ui.element('objectCount').textContent, 'unavailable');
    // The provider's reason stays literal text.
    assert.equal(ui.element('objectsUnavailable').textContent, `Object state unavailable: ${markup}`);
    assert.doesNotMatch(ui.element('objects').innerHTML, /<img/);
    assert.equal(ui.run('globalThis.injected'), undefined);

    // A catalogue refresh finds no stale objects to describe.
    await ui.run('refreshAssets()');
    assert.equal(ui.element('objectDescription0'), undefined);
});

test('every scene edit reads the runtime objects again once its goal completes', async () => {
    const ui = await page([block], [redBlock]);
    const since = ui.requests.length;
    await ui.run('clearScene()');
    await ui.run('spawnObject()');
    await ui.run('moveObject(0)');
    await ui.run('removeObject(0)');
    assert.deepEqual(ui.requests.slice(since).map(r => r.path), [
        '/api/scene/clear', '/api/objects',
        '/api/objects/spawn', '/api/objects',
        '/api/objects/move', '/api/objects',
        '/api/objects/remove', '/api/objects',
    ]);
    assert.equal(ui.requests[since + 2].body.asset_id, block.asset_id);
    assert.equal(ui.requests[since + 2].body.yaw, 0, 'a spawn stands as authored unless the form names a yaw');
    assert.deepEqual(ui.requests[since + 4].body, { object_id: 'obj_1', position: [0.5, 0, 0.8] });
    assert.deepEqual(ui.requests[since + 6].body, { object_id: 'obj_1' });
});

// Resolves once the page has handled every line pushed to `stream` and waits
// for more, or once `running` settled, whichever comes first.
const handled = (stream, running) => Promise.race([stream.drained(), running]);

// How the node answers the cancel of a load: taken by the simulator, or not
// delivered to it.
const CANCEL_TAKEN = 'the simulator answered the cancel of load_scene: SIGNALLED';
const CANCEL_LOST = "the cancel of load_scene was not delivered: service 'cancel' timed out";

test('a scene load shows its progress line by line and refreshes only once it succeeded', async () => {
    const ui = await page([staticScene], [], { capabilities: { lighting: true }, lighting: warehouseLighting });
    const since = ui.requests.length;
    const stream = ui.stream('/api/scene/load');
    const loading = ui.run('loadScene()');

    await handled(stream, loading);
    assert.equal(ui.element('status').textContent, 'Loading scene...');
    assert.equal(ui.element('cancelLoad').hidden, true, 'nothing to cancel before the simulator accepts the load');

    stream.push({ goal: '7' });
    await handled(stream, loading);
    assert.equal(ui.element('cancelLoad').hidden, false);

    stream.push({ progress: { bytes_fetched: 123_400_000, files_ready: 41, building: false } });
    await handled(stream, loading);
    assert.equal(ui.element('status').textContent, 'Loading scene: 123.4 MB, 41 files');

    stream.push({ progress: { bytes_fetched: 368_000_000, files_ready: 135, building: true } });
    await handled(stream, loading);
    assert.equal(ui.element('status').textContent, 'Building the scene');
    assert.deepEqual(ui.requests.slice(since).map(r => r.path), ['/api/scene/load'],
        'nothing is read again while the load runs');

    stream.push({ success: true, message: 'Loaded scene/warehouse_static' });
    stream.end();
    await loading;
    assert.deepEqual(ui.requests.slice(since).map(r => r.path), ['/api/scene/load', '/api/objects', '/api/lighting']);
    assert.equal(ui.element('status').textContent, 'Loaded scene/warehouse_static');
    assert.equal(ui.element('status').style.borderColor, '#303741');
    assert.equal(ui.element('cancelLoad').hidden, true);
});

test('a scene load that fails once accepted shows the reason and reads nothing again', async () => {
    const ui = await page();
    const since = ui.requests.length;
    const stream = ui.stream('/api/scene/load');
    stream.push({ goal: '2' }, { success: false, message: 'load_scene made no progress for 60 s; it was cancelled', cancelled_by: 'commander' });
    stream.end();

    await ui.run('loadScene()');
    assert.deepEqual(ui.requests.slice(since).map(r => r.path), ['/api/scene/load']);
    assert.equal(ui.element('status').textContent, 'load_scene made no progress for 60 s; it was cancelled');
    assert.equal(ui.element('status').style.borderColor, '#9b424c');
    assert.equal(ui.element('cancelLoad').hidden, true);
});

test('a scene load refused before it starts shows the refusal', async () => {
    const ui = await page();
    const since = ui.requests.length;
    ui.refuse('/api/scene/load', 400, 'load_scene rejected: the simulator is shutting down');

    await ui.run('loadScene()');
    assert.deepEqual(ui.requests.slice(since).map(r => r.path), ['/api/scene/load']);
    assert.equal(ui.element('status').textContent, 'load_scene rejected: the simulator is shutting down');
    assert.equal(ui.element('status').style.borderColor, '#9b424c');
});

test('a scene load the simulator cancelled shows the reason the simulator gave', async () => {
    const ui = await page();
    const since = ui.requests.length;
    const stream = ui.stream('/api/scene/load');
    const message = 'load_scene was cancelled by the simulator: '
        + 'load_scene(scene/warehouse_static) was replaced by load_scene(scene/warehouse_dynamic)';
    stream.push(
        { goal: '3' },
        { progress: { bytes_fetched: 1_000_000, files_ready: 2, building: false } },
        { success: false, message, cancelled_by: 'simulator' },
    );
    stream.end();

    await ui.run('loadScene()');
    assert.equal(ui.element('status').textContent, message);
    assert.equal(ui.element('status').style.borderColor, '#9b424c');
    assert.deepEqual(ui.requests.slice(since).map(r => r.path), ['/api/scene/load']);
    assert.equal(ui.element('cancelLoad').hidden, true);
});

test('a newer load refused before it starts leaves the running load its status, its cancel and its end', async () => {
    const ui = await page([staticScene], [], { capabilities: { lighting: true }, lighting: warehouseLighting });
    const since = ui.requests.length;
    const running = ui.stream('/api/scene/load');
    const runningLoad = ui.run('loadScene()');
    running.push({ goal: '1' }, { progress: { bytes_fetched: 1_000_000, files_ready: 1, building: false } });
    await handled(running, runningLoad);

    ui.refuse('/api/scene/load', 400, 'load_scene rejected: the simulator admits one load at a time');
    await ui.run('loadScene()');
    assert.equal(ui.element('status').textContent, 'load_scene rejected: the simulator admits one load at a time');
    assert.equal(ui.element('status').style.borderColor, '#9b424c');
    assert.equal(ui.element('cancelLoad').hidden, false, 'the running load can still be cancelled');

    running.push({ progress: { bytes_fetched: 2_000_000, files_ready: 2, building: false } });
    await handled(running, runningLoad);
    assert.equal(ui.element('status').textContent, 'Loading scene: 2.0 MB, 2 files');

    running.push({ success: true, message: 'Loaded scene/warehouse_static' });
    running.end();
    await runningLoad;
    assert.equal(ui.element('status').textContent, 'Loaded scene/warehouse_static');
    assert.equal(ui.element('cancelLoad').hidden, true);
    assert.deepEqual(ui.requests.slice(since).map(r => r.path), [
        '/api/scene/load', '/api/scene/load', '/api/objects', '/api/lighting',
    ]);
});

test('a load this page replaced ends in silence and leaves the status to the newer load', async () => {
    const ui = await page();
    const first = ui.stream('/api/scene/load');
    const second = ui.stream('/api/scene/load');
    const firstLoad = ui.run('loadScene()');
    first.push({ goal: '1' });
    await handled(first, firstLoad);

    const secondLoad = ui.run('loadScene()');
    second.push({ goal: '2' }, { progress: { bytes_fetched: 0, files_ready: 5, building: false } });
    await handled(second, secondLoad);
    first.push({ success: false, message: 'load_scene was cancelled by the simulator', cancelled_by: 'simulator' });
    first.end();
    await firstLoad;
    assert.equal(ui.element('status').textContent, 'Loading scene: 0.0 MB, 5 files');
    assert.equal(ui.element('cancelLoad').hidden, false, 'the newer load can still be cancelled');

    ui.answer('/api/goals/2/cancel', { message: CANCEL_TAKEN });
    const cancelling = ui.click('cancelLoad');
    assert.equal(second.aborted, true);
    await cancelling;
    await secondLoad;
    assert.deepEqual(ui.requests.at(-1), { path: '/api/goals/2/cancel', body: {} });
    assert.equal(ui.element('status').textContent, CANCEL_TAKEN);
});

test('the cancel button aborts the page\'s stream of the load, then cancels the load by the goal id it named', async () => {
    const ui = await page();
    const since = ui.requests.length;
    const stream = ui.stream('/api/scene/load');
    const loading = ui.run('loadScene()');
    stream.push({ goal: '7' });
    await handled(stream, loading);

    ui.answer('/api/goals/7/cancel', { message: CANCEL_TAKEN });
    const cancelling = ui.click('cancelLoad');
    assert.equal(stream.aborted, true);
    assert.equal(ui.element('status').textContent, 'Cancelling the scene load...');
    assert.equal(ui.element('cancelLoad').hidden, true);

    await cancelling;
    await loading;
    assert.deepEqual(ui.requests.slice(since), [
        { path: '/api/scene/load', body: { asset_id: staticScene.asset_id, scale: 1 } },
        { path: '/api/goals/7/cancel', body: {} },
    ]);
    // With its stream gone, the page says how the cancel was answered, and
    // the aborted stream says nothing.
    assert.equal(ui.element('status').textContent, CANCEL_TAKEN);
    assert.equal(ui.element('status').style.borderColor, '#303741');
});

test('a page whose six connections all carry streams still sends the cancel of its load', async () => {
    const ui = await page([staticScene, block], []);
    const stream = ui.stream('/api/scene/load');
    const loading = ui.run('loadScene()');
    stream.push({ goal: '7' });
    await handled(stream, loading);
    for (const goal of ['8', '9', '10', '11', '12']) {
        const spawn = ui.stream('/api/objects/spawn');
        const spawning = ui.run('spawnObject()');
        spawn.push({ goal });
        await handled(spawn, spawning);
    }

    ui.answer('/api/goals/7/cancel', { message: CANCEL_TAKEN });
    const cancelling = ui.click('cancelLoad');
    assert.deepEqual(ui.requests.at(-1), { path: '/api/goals/7/cancel', body: {} },
        'the cancel goes out on the connection the aborted stream freed, while five spawns hold theirs');
    assert.equal(stream.aborted, true);

    await cancelling;
    await loading;
    assert.equal(ui.element('status').textContent, CANCEL_TAKEN);
});

test('a cancel answered after a newer load was accepted leaves the status to the newer load', async () => {
    const ui = await page();
    const first = ui.stream('/api/scene/load');
    const second = ui.stream('/api/scene/load');
    const firstLoad = ui.run('loadScene()');
    first.push({ goal: '7' });
    await handled(first, firstLoad);

    const refuseCancel = ui.refuseLater('/api/goals/7/cancel');
    const cancelling = ui.click('cancelLoad');
    assert.equal(first.aborted, true);
    await firstLoad;

    const secondLoad = ui.run('loadScene()');
    second.push({ goal: '8' }, { progress: { bytes_fetched: 3_000_000, files_ready: 4, building: false } });
    await handled(second, secondLoad);
    refuseCancel(502, CANCEL_LOST);
    await cancelling;
    assert.equal(ui.element('status').textContent, 'Loading scene: 3.0 MB, 4 files');
    assert.equal(ui.element('status').style.borderColor, '#303741');
    assert.equal(ui.element('cancelLoad').hidden, false, 'the newer load can be cancelled');

    second.push({ success: true, message: 'Loaded scene/warehouse_static' });
    second.end();
    await secondLoad;
    assert.equal(ui.element('status').textContent, 'Loaded scene/warehouse_static');
});

test('a cancel the node refuses shows the refusal', async () => {
    const ui = await page();
    const stream = ui.stream('/api/scene/load');
    const loading = ui.run('loadScene()');
    stream.push({ goal: '7' });
    await handled(stream, loading);

    ui.refuse('/api/goals/7/cancel', 502, CANCEL_LOST);
    const cancelling = ui.click('cancelLoad');
    assert.equal(stream.aborted, true);
    await cancelling;
    await loading;
    assert.equal(ui.element('status').textContent, CANCEL_LOST);
    assert.equal(ui.element('status').style.borderColor, '#9b424c');
    assert.equal(ui.element('cancelLoad').hidden, true);
});

test('the stream reader takes a line split over two reads and two lines in one read', async () => {
    const ui = await page();
    const stream = ui.stream('/api/scene/load');
    const loading = ui.run('loadScene()');

    stream.pushText('{"goal": "4"}\n{"progress": {"bytes_fetched": 1500000, "fi');
    await handled(stream, loading);
    assert.equal(ui.element('cancelLoad').hidden, false);
    assert.equal(ui.element('status').textContent, 'Loading scene...');

    stream.pushText('les_ready": 2, "building": false}}\n');
    await handled(stream, loading);
    assert.equal(ui.element('status').textContent, 'Loading scene: 1.5 MB, 2 files');

    // The last line needs no newline before the stream ends.
    stream.pushText('{"success": true, "message": "Loaded"}');
    stream.end();
    await loading;
    assert.equal(ui.element('status').textContent, 'Loaded');
});

test('a stream cut before its result is reported as such', async () => {
    const ui = await page();
    const since = ui.requests.length;
    ui.stream('/api/scene/load').end();

    await ui.run('loadScene()');
    assert.equal(ui.element('status').textContent, '/api/scene/load ended without a result');
    assert.equal(ui.element('status').style.borderColor, '#9b424c');
    assert.deepEqual(ui.requests.slice(since).map(r => r.path), ['/api/scene/load']);
});

test('a spawn shows its progress and reads the objects again once it succeeded', async () => {
    const ui = await page([block], []);
    const since = ui.requests.length;
    const stream = ui.stream('/api/objects/spawn');
    const spawning = ui.run('spawnObject()');

    stream.push({ goal: '5' }, { progress: { bytes_fetched: 1_200_000, files_ready: 3, building: false } });
    await handled(stream, spawning);
    assert.equal(ui.element('status').textContent, `Loading ${block.asset_id}: 1.2 MB, 3 files`);

    stream.push({ progress: { bytes_fetched: 1_200_000, files_ready: 4, building: true } });
    await handled(stream, spawning);
    assert.equal(ui.element('status').textContent, `Building ${block.asset_id}`);
    assert.deepEqual(ui.requests.slice(since).map(r => r.path), ['/api/objects/spawn']);

    stream.push({ success: true, message: `Spawned ${block.asset_id}`, object_id: 'obj_9' });
    stream.end();
    await spawning;
    assert.deepEqual(ui.requests.slice(since).map(r => r.path), ['/api/objects/spawn', '/api/objects']);
    assert.equal(ui.element('status').textContent, `Spawned ${block.asset_id}\nobject_id=obj_9`);
});

test('a spawn that fails once accepted shows the reason and reads nothing again', async () => {
    const ui = await page([block], []);
    const since = ui.requests.length;
    const stream = ui.stream('/api/objects/spawn');
    stream.push({ goal: '6' }, { success: false, message: 'spawn_object ended: the simulator is gone' });
    stream.end();

    await ui.run('spawnObject()');
    assert.deepEqual(ui.requests.slice(since).map(r => r.path), ['/api/objects/spawn']);
    assert.equal(ui.element('status').textContent, 'spawn_object ended: the simulator is gone');
    assert.equal(ui.element('status').style.borderColor, '#9b424c');
});

// What a launch may bind beside the scene, as the node reports it.
const sun = {
    id: 'sun', kind: 'directional', label: 'Sun',
    properties: {
        illuminance_lux: { unit: 'lx', min: 0, max: 200000, default: 1000, value: 1200 },
        color_rgb: { unit: 'linear rgb', default: [1, 1, 1], value: [1, 0.95, 0.9] },
        direction: { unit: 'unit vector', default: [0, 0, -1], value: [0.3, 0, -0.95] },
    },
};
const lamp = {
    id: 'lamp', kind: 'spot', label: 'Desk lamp',
    properties: {
        luminous_flux_lm: { unit: 'lm', min: 0, max: 20000, default: 800, value: 800 },
        color_rgb: { unit: 'linear rgb', default: [1, 1, 1], value: [1, 1, 1] },
        position_m: { unit: 'm', default: [0, 0, 2], value: [0.5, 0.2, 1.8] },
        direction: { unit: 'unit vector', default: [0, 0, -1], value: [0, 0, -1] },
        inner_angle_rad: { unit: 'rad', min: 0, max: 1.5, default: 0.3, value: 0.3 },
        outer_angle_rad: { unit: 'rad', min: 0, max: 1.5, default: 0.5, value: 0.5 },
    },
};
const sky = {
    id: 'sky', kind: 'environment', label: 'Sky',
    properties: {
        radiance_scale: { unit: 'ratio', min: 0, max: 10, default: 1, value: 1 },
        orientation_xyzw: { unit: 'quaternion', default: [0, 0, 0, 1], value: [0, 0, 0, 1] },
    },
};
const warehouseLighting = { scene: staticScene.asset_id, ambient_lux: 120, fill_lux: 40, lights: [sun, lamp, sky] };
const steel = {
    id: 'steel', label: 'Brushed steel',
    scope: { owner: 'scene', surfaces: 3, bodies: ['shelf_1', 'shelf_2'] },
    properties: {
        color_rgb: { unit: 'linear rgb', default: [0.6, 0.6, 0.6], value: [0.6, 0.6, 0.6] },
        metallic: { unit: 'ratio', min: 0, max: 1, default: 1, value: 1 },
        roughness: { unit: 'ratio', min: 0, max: 1, default: 0.3, value: 0.3 },
    },
};
const wrist = {
    id: 'alpha/wrist_left', robot: 'alpha', camera: 'wrist_left', kind: 'rgb',
    info: { width: 1280, height: 720, frames_per_second: 30, encoding: 'rgb8' },
    profile: {
        device: 'c920', label: 'Logitech C920', encoding: 'mjpeg',
        controls: {
            exposure: {
                supported: true, modes: ['auto', 'manual'], unit: 'us', min: 100, max: 33000,
                default_mode: 'auto', default_value: 8000, mode: 'auto', value: 8300,
            },
            gain: { supported: false },
            brightness: {
                supported: true, modes: ['manual'], unit: 'level', min: 0, max: 255,
                default_mode: 'manual', default_value: 128, mode: 'manual', value: 128,
            },
        },
    },
    profile_message: 'profile of Logitech C920',
};
const chest = {
    id: 'alpha/chest', robot: 'alpha', camera: 'chest', kind: 'rgbd',
    info: { width: 640, height: 480, frames_per_second: 15, encoding: 'rgb8' },
    profile: null,
    profile_message: 'the simulation describes no profile for alpha/chest',
};
const cards = ['lightingCard', 'materialsCard', 'camerasCard'];

test('with only scene manipulation bound the capability cards stay hidden and nothing polls', async () => {
    const ui = await page();
    assert.deepEqual(ui.requests.map(r => r.path), ['/api/capabilities', '/api/assets', '/api/objects', '/api/robots']);
    assert.deepEqual(ui.intervals, []);
    for (const id of cards) {
        assert.equal(ui.element(id).hidden, true, `#${id} stays hidden`);
    }
    // A scene edit reads nothing a vacant capability would answer.
    await ui.run('loadScene()');
    assert.deepEqual(ui.requests.slice(-2).map(r => r.path), ['/api/scene/load', '/api/objects']);
});

test('the capability cards open hidden before any script runs', () => {
    const elements = new Map();
    registerElements(html, elements);
    for (const id of cards) {
        assert.equal(elements.get(id).hidden, true, `#${id} is hidden in the page source`);
    }
});

test('bound lighting renders one control per listed property and applies a route as one request', async () => {
    const ui = await page([staticScene], [], { capabilities: { lighting: true }, lighting: warehouseLighting });
    assert.deepEqual(ui.requests.map(r => r.path),
        ['/api/capabilities', '/api/assets', '/api/objects', '/api/robots', '/api/lighting']);
    assert.deepEqual(ui.intervals, [3000]);
    assert.equal(ui.element('lightingCard').hidden, false);
    assert.equal(ui.element('materialsCard').hidden, true);
    assert.equal(ui.element('camerasCard').hidden, true);
    assert.equal(ui.element('lightingSummary').textContent,
        `${staticScene.asset_id} | ambient 120 lx | fill 40 lx`);

    // A control per property the light lists, and none for the rest.
    assert.equal(ui.element('lighting0_illuminance_lux_0').value, '1200');
    assert.equal(ui.element('lighting0_illuminance_lux_0').attributes.min, '0');
    assert.equal(ui.element('lighting0_illuminance_lux_0').attributes.max, '200000');
    assert.equal(ui.element('lighting0_color_rgb_2').value, '0.9');
    assert.equal(ui.element('lighting0_direction_2').value, '-0.95');
    assert.equal(ui.element('lighting0_position_m_0'), undefined, 'a directional light has no position');
    assert.equal(ui.element('lighting0_inner_angle_rad_0'), undefined, 'a directional light has no cone');
    assert.equal(ui.element('lighting1_position_m_2').value, '1.8');
    assert.equal(ui.element('lighting1_outer_angle_rad_0').value, '0.5');
    assert.equal(ui.element('lighting2_orientation_xyzw_3').value, '1');
    assert.equal(ui.element('lighting2_color_rgb_0'), undefined, 'an environment light has no colour');
    assert.equal(ui.element('lighting0_intensity_current').textContent, 'illuminance_lux 1200 (default 1000)');
    assert.equal(ui.element('lighting0_color_current').textContent,
        'color_rgb [1, 0.95, 0.9] (default [1, 1, 1])');
    assert.equal(ui.element('lighting1_cone_current').textContent,
        'inner_angle_rad 0.3 (default 0.3) | outer_angle_rad 0.5 (default 0.5)');
    assert.match(ui.element('lightingTargets').innerHTML, /spot \| lamp/);

    ui.element('lighting1_inner_angle_rad_0').value = '0.2';
    ui.element('lighting1_outer_angle_rad_0').value = '0.6';
    await ui.run("applyProperty('lighting', 1, 'cone')");
    assert.deepEqual(ui.requests.slice(-2).map(r => r.path), ['/api/lighting/cone', '/api/lighting']);
    assert.deepEqual(ui.requests.at(-2).body, { light_id: 'lamp', inner_angle: 0.2, outer_angle: 0.6 });

    ui.element('lighting0_color_rgb_0').value = '0.5';
    await ui.run("applyProperty('lighting', 0, 'color')");
    assert.deepEqual(ui.requests.at(-2).body, { light_id: 'sun', color: [0.5, 0.95, 0.9] });
    assert.equal(ui.element('lighting0_color_rgb_0').value, '0.5', 'an edit survives the read that follows');

    ui.element('lighting2_orientation_xyzw_0').value = '0.7';
    await ui.run("applyProperty('lighting', 2, 'orientation')");
    assert.deepEqual(ui.requests.at(-2).body, { light_id: 'sky', orientation: [0.7, 0, 0, 1] });

    await ui.run("resetPanel('lighting')");
    assert.deepEqual(ui.requests.slice(-2).map(r => r.path), ['/api/lighting/reset', '/api/lighting']);
    assert.deepEqual(ui.requests.at(-2).body, {});
    assert.equal(ui.element('status').textContent, 'Scene loaded');
});

test('a refresh shows the values the provider reports and rebuilds the controls only when its lights change', async () => {
    const ui = await page([staticScene], [], { capabilities: { lighting: true }, lighting: warehouseLighting });
    ui.element('lighting0_illuminance_lux_0').value = '3000';
    const brighter = { ...sun.properties.illuminance_lux, value: 5000 };
    ui.setLighting({ ...warehouseLighting, lights: [{ ...sun, properties: { ...sun.properties, illuminance_lux: brighter } }, lamp, sky] });
    await ui.run('refreshPanels()');
    assert.equal(ui.element('lighting0_intensity_current').textContent, 'illuminance_lux 5000 (default 1000)');
    assert.equal(ui.element('lighting0_illuminance_lux_0').value, '3000', 'the edit in progress is kept');

    ui.setLighting({ ...warehouseLighting, lights: [sun] });
    await ui.run('refreshPanels()');
    assert.equal(ui.element('lighting1_cone_current'), undefined, 'a light that is gone takes its controls with it');
    assert.equal(ui.element('lighting0_illuminance_lux_0').value, '1200');

    ui.setLighting(null);
    await ui.run('refreshPanels()');
    assert.equal(ui.element('lightingCard').hidden, true);
    assert.equal(ui.element('status').textContent, 'Loaded 1 assets', 'a periodic read that fails is silent');

    ui.setLighting(warehouseLighting);
    await ui.run('refreshPanels()');
    assert.equal(ui.element('lightingCard').hidden, false);
    assert.equal(ui.element('lighting1_cone_current').textContent,
        'inner_angle_rad 0.3 (default 0.3) | outer_angle_rad 0.5 (default 0.5)');
});

test('a scene load reads the capability panels again once its goal completes', async () => {
    const ui = await page([staticScene], [], {
        capabilities: { lighting: true, materials: true }, lighting: warehouseLighting, materials: { materials: [steel] },
    });
    const since = ui.requests.length;
    await ui.run('loadScene()');
    await ui.run('clearScene()');
    assert.deepEqual(ui.requests.slice(since).map(r => r.path), [
        '/api/scene/load', '/api/objects', '/api/lighting', '/api/materials',
        '/api/scene/clear', '/api/objects', '/api/lighting', '/api/materials',
    ]);
});

test('bound materials show their scope and defaults and post colour and finish by material', async () => {
    const ui = await page([staticScene], [], { capabilities: { materials: true }, materials: { materials: [steel] } });
    assert.equal(ui.element('materialsCard').hidden, false);
    assert.equal(ui.element('lightingCard').hidden, true);
    assert.equal(ui.element('materialsSummary').textContent, '1 materials');
    assert.match(ui.element('materialsTargets').innerHTML, /shared by 3 surfaces on: shelf_1, shelf_2/);
    assert.equal(ui.element('materials0_color_current').textContent,
        'color_rgb [0.6, 0.6, 0.6] (default [0.6, 0.6, 0.6])');
    assert.equal(ui.element('materials0_finish_current').textContent,
        'metallic 1 (default 1) | roughness 0.3 (default 0.3)');
    assert.equal(ui.element('materials0_metallic_0').attributes.max, '1');

    ui.element('materials0_metallic_0').value = '0.5';
    await ui.run("applyProperty('materials', 0, 'finish')");
    assert.deepEqual(ui.requests.at(-2),
        { path: '/api/materials/finish', body: { material_id: 'steel', metallic: 0.5, roughness: 0.3 } });
    ui.element('materials0_color_rgb_2').value = '0.9';
    await ui.run("applyProperty('materials', 0, 'color')");
    assert.deepEqual(ui.requests.at(-2),
        { path: '/api/materials/color', body: { material_id: 'steel', color: [0.6, 0.6, 0.9] } });
    await ui.run("resetPanel('materials')");
    assert.deepEqual(ui.requests.slice(-2).map(r => r.path), ['/api/materials/reset', '/api/materials']);
});

test('bound cameras render the controls their profile supports and route by robot and camera', async () => {
    const ui = await page([staticScene], [], { capabilities: { cameras: true }, cameras: [wrist, chest] });
    assert.deepEqual(ui.requests.map(r => r.path),
        ['/api/capabilities', '/api/assets', '/api/objects', '/api/robots', '/api/cameras']);
    assert.equal(ui.element('camerasCard').hidden, false);
    assert.equal(ui.element('camera0_info').textContent, 'rgb | 1280x720 @ 30 fps | rgb8');
    assert.equal(ui.element('camera0_profile').textContent, 'profile of Logitech C920');
    assert.ok(ui.element('camera0_exposure_0'), 'a supported control has its input');
    assert.ok(ui.element('camera0_exposure_mode'), 'exposure has its mode');
    assert.equal(ui.element('camera0_exposure_0').attributes.max, '33000');
    assert.equal(ui.element('camera0_exposure_0').value, '8300');
    assert.ok(ui.element('camera0_brightness_0'));
    assert.equal(ui.element('camera0_brightness_mode'), undefined, 'a level has no mode');
    assert.equal(ui.element('camera0_gain_0'), undefined, 'an unsupported control is not rendered');
    assert.equal(ui.element('camera0_contrast_0'), undefined, 'a control the profile omits is not rendered');
    assert.equal(ui.element('camera0_exposure_current').textContent, 'auto 8300 us (default auto 8000)');
    assert.equal(ui.element('camera0_brightness_current').textContent, '128 level (default 128)');
    assert.match(ui.element('cameraTargets').innerHTML, /resetCamera\(0\)/);

    // A camera without a profile lists its stream and says why it has no controls.
    assert.equal(ui.element('camera1_info').textContent, 'rgbd | 640x480 @ 15 fps | rgb8');
    assert.equal(ui.element('camera1_profile').textContent, chest.profile_message);
    assert.equal(ui.element('camera1_exposure_0'), undefined);
    assert.doesNotMatch(ui.element('cameraTargets').innerHTML, /resetCamera\(1\)/);

    ui.element('camera0_exposure_mode').value = 'manual';
    ui.element('camera0_exposure_0').value = '8000';
    await ui.run("applyCamera(0, 'exposure')");
    assert.deepEqual(ui.requests.slice(-2).map(r => r.path), ['/api/cameras/alpha/wrist_left/exposure', '/api/cameras']);
    assert.deepEqual(ui.requests.at(-2).body, { value: 8000, mode: 'manual' });

    ui.element('camera0_brightness_0').value = '200';
    await ui.run("applyCamera(0, 'brightness')");
    assert.deepEqual(ui.requests.at(-2), { path: '/api/cameras/alpha/wrist_left/brightness', body: { value: 200 } });

    await ui.run('resetCamera(0)');
    assert.deepEqual(ui.requests.slice(-2).map(r => r.path), ['/api/cameras/alpha/wrist_left/reset', '/api/cameras']);
});

test('provider text in the capability cards stays literal', async () => {
    const markup = '<img src=x onerror="globalThis.injected = true">';
    const ui = await page([staticScene], [], {
        capabilities: { lighting: true, materials: true, cameras: true },
        lighting: { ...warehouseLighting, lights: [{ ...sun, label: markup, kind: markup }] },
        materials: { materials: [{ ...steel, label: markup, scope: { ...steel.scope, bodies: [markup] } }] },
        cameras: [{ ...wrist, id: markup, profile_message: markup }],
    });
    for (const id of ['lightingTargets', 'materialsTargets', 'cameraTargets']) {
        assert.doesNotMatch(ui.element(id).innerHTML, /<img/, `#${id} keeps provider text literal`);
    }
    assert.equal(ui.element('camera0_profile').textContent, markup);
    assert.equal(ui.run('globalThis.injected'), undefined);
});
