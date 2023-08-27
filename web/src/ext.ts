import {IWidget, widgetTypes} from "/types/litegraph";
import {initEditor} from "./editor";
import {MONACO_WIDGET} from "./widgets";

// @ts-ignore
self.MonacoEnvironment = {
    baseUrl: 'extensions/ClipStuff/',
    getWorkerUrl: function (moduleId, label) {
        if (label === 'json') {
            return '/extensions/ClipStuff/json.worker.bundle.js';
        }
        if (label === 'css' || label === 'scss' || label === 'less') {
            return './extensions/ClipStuff/css.worker.bundle.js';
        }
        if (label === 'html' || label === 'handlebars' || label === 'razor') {
            return './extensions/ClipStuff/html.worker.bundle.js';
        }
        if (label === 'typescript' || label === 'javascript') {
            return './extensions/ClipStuff/ts.worker.bundle.js';
        }
        return './extensions/ClipStuff/editor.worker.bundle.js';
    }
};

let isSetup = false;

const app = await import(/* webpackIgnore: true */'/scripts/app.js')
app.app.registerExtension({
    name: "PromptLang.textNode",
    async setup(app) {
        if (isSetup) return;
        initEditor();
        isSetup = true;
    },
    async getCustomWidgets(app): Record<string, (node, inputName, inputData, app) => { widget?: IWidget; minWidth?: number; minHeight?: number }> {
        if (!isSetup) {
            initEditor();
            isSetup = true;
        }

        return {
            MONACO: MONACO_WIDGET
        }
    },
})
    // resolveAppLoading();
// })
// await appLoadingPromise;
