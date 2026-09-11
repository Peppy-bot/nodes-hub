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
        this.style = {};
        this.children = [];
    }

    replaceChildren(...children) {
        this.children = children;
        if (this.tag === 'select') this.value = children[0]?.value || '';
    }

    set innerHTML(value) {
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

async function page(catalogue = [staticScene, dynamicScene, cage], objects = []) {
    const elements = new Map();
    registerElements(html, elements);
    const requests = [];
    let currentCatalogue = catalogue;
    const context = vm.createContext({
        Option,
        document: { getElementById: id => {
            assert.ok(elements.has(id), `page contains #${id}`);
            return elements.get(id);
        } },
        fetch: async (path, options) => {
            requests.push({ path, body: options.body && JSON.parse(options.body) });
            const data = path === '/api/assets' ? { assets: currentCatalogue }
                : path === '/api/objects' ? { objects }
                : { message: 'Scene loaded' };
            return { ok: true, json: async () => ({ success: true, ...data }) };
        },
    });
    await vm.runInContext(script, context);
    const element = id => elements.get(id);
    return {
        element, requests,
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
    };
}

test('fetched scenes expose their catalogue names and selected descriptions', async () => {
    const ui = await page();
    assert.deepEqual(ui.requests.map(r => r.path), ['/api/assets', '/api/objects']);
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
    assert.deepEqual(ui.requests.at(-1), {
        path: '/api/scene/load', body: { asset_id: dynamicScene.asset_id, scale: 1.5 },
    });
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
