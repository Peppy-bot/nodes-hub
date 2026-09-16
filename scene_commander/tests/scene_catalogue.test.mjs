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
// lists.
async function page(catalogue = [staticScene, dynamicScene, cage], objects = [],
    { unavailable = 0, objectState = null, capabilities = {}, lighting = null, materials = null, cameras = [] } = {}) {
    const elements = new Map();
    registerElements(html, elements);
    const requests = [];
    const waits = [];
    const intervals = [];
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
    const context = vm.createContext({
        Option,
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
        fetch: async (path, options) => {
            requests.push({ path, body: options.body && JSON.parse(options.body) });
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
                : path === '/api/capabilities'
                    ? { lighting: false, materials: false, cameras: [], ...capabilities }
                : path === '/api/lighting' ? { lighting: currentLighting }
                : path === '/api/materials' ? { materials: currentMaterials }
                : path === '/api/cameras' ? { cameras: currentCameras, count: currentCameras.length }
                : { message: 'Scene loaded' };
            return { ok: true, json: async () => ({ success: true, ...data }) };
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
    };
}

test('fetched scenes expose their catalogue names and selected descriptions', async () => {
    const ui = await page();
    assert.deepEqual(ui.requests.map(r => r.path), ['/api/capabilities', '/api/assets', '/api/objects']);
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
        ['/api/capabilities', '/api/assets', '/api/assets', '/api/assets', '/api/objects']);
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
    assert.deepEqual(ui.requests.map(r => r.path), ['/api/capabilities', '/api/assets', '/api/objects']);
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
    assert.deepEqual(ui.requests[since + 4].body, { object_id: 'obj_1', position: [0.5, 0, 0.8] });
    assert.deepEqual(ui.requests[since + 6].body, { object_id: 'obj_1' });
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
    id: 'alpha_wrist_left', kind: 'rgb',
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
    id: 'alpha_chest', kind: 'rgbd',
    info: { width: 640, height: 480, frames_per_second: 15, encoding: 'rgb8' },
    profile: null,
    profile_message: 'no camera profile is bound for alpha_chest in this launch',
};
const cards = ['lightingCard', 'materialsCard', 'camerasCard'];

test('with only scene manipulation bound the capability cards stay hidden and nothing polls', async () => {
    const ui = await page();
    assert.deepEqual(ui.requests.map(r => r.path), ['/api/capabilities', '/api/assets', '/api/objects']);
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
        ['/api/capabilities', '/api/assets', '/api/objects', '/api/lighting']);
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

test('bound cameras render the controls their profile supports and route by camera id', async () => {
    const ui = await page([staticScene], [], {
        capabilities: { cameras: [{ id: wrist.id, kind: 'rgb', profile: true }, { id: chest.id, kind: 'rgbd', profile: false }] },
        cameras: [wrist, chest],
    });
    assert.deepEqual(ui.requests.map(r => r.path),
        ['/api/capabilities', '/api/assets', '/api/objects', '/api/cameras']);
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
    assert.deepEqual(ui.requests.slice(-2).map(r => r.path), ['/api/cameras/alpha_wrist_left/exposure', '/api/cameras']);
    assert.deepEqual(ui.requests.at(-2).body, { value: 8000, mode: 'manual' });

    ui.element('camera0_brightness_0').value = '200';
    await ui.run("applyCamera(0, 'brightness')");
    assert.deepEqual(ui.requests.at(-2), { path: '/api/cameras/alpha_wrist_left/brightness', body: { value: 200 } });

    await ui.run('resetCamera(0)');
    assert.deepEqual(ui.requests.slice(-2).map(r => r.path), ['/api/cameras/alpha_wrist_left/reset', '/api/cameras']);
});

test('provider text in the capability cards stays literal', async () => {
    const markup = '<img src=x onerror="globalThis.injected = true">';
    const ui = await page([staticScene], [], {
        capabilities: { lighting: true, materials: true, cameras: [{ id: markup, kind: 'rgb', profile: true }] },
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
