import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

// Execute the page served by the Python node. This harness supplies only the
// DOM and HTTP boundary; selection, rendering and requests use the real script.
const source = readFileSync(new URL('../src/openarm_scene_commander/__main__.py', import.meta.url), 'utf8');
const html = source.match(/HTML = r"""([\s\S]*?)"""/)[1];
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

class Option {
    constructor(text, value) {
        this.textContent = String(text);
        this.value = String(value);
    }
}

class Element {
    constructor(tag, attributes) {
        this.tag = tag;
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
        this.replaceChildren(...Array.from(value.matchAll(/<option value="([^"]*)">([\s\S]*?)<\/option>/g),
            ([, id, name]) => new Option(name.trim(), id)));
    }
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
};

async function page(catalogue = [staticScene, dynamicScene, cage]) {
    const elements = new Map();
    for (const [, tag, raw] of html.matchAll(/<([a-z]+)\b([^>]*\bid="[^>]+)>/g)) {
        const attributes = Object.fromEntries(Array.from(raw.matchAll(/([\w-]+)(?:="([^"]*)")?/g),
            ([, name, value]) => [name, value || '']));
        elements.set(attributes.id, new Element(tag, attributes));
    }
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
                : path === '/api/objects' ? { objects: [] }
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
