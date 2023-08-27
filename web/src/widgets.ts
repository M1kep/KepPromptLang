import {widgetTypes} from "/types/litegraph";
import * as monaco from 'monaco-editor';
export const MONACO_WIDGET = (node, inputName, inputData, app) => {
    const MIN_SIZE = 50;

    function computeSize(size) {
        if (node.widgets[0].last_y == null) return;

        let y = node.widgets[0].last_y;
        let freeSpace = size[1] - y;

        // Compute the height of all non customtext widgets
        let widgetHeight = 0;
        const multi = [];
        for (let i = 0; i < node.widgets.length; i++) {
            const w = node.widgets[i];
            if (w.type === "MONACO") {
                multi.push(w);
            } else {
                if (w.computeSize) {
                    widgetHeight += w.computeSize()[1] + 4;
                } else {
                    widgetHeight += LiteGraph.NODE_WIDGET_HEIGHT + 4;
                }
            }
        }

        // See how large each text input can be
        freeSpace -= widgetHeight;
        freeSpace /= multi.length + (!!node.imgs?.length);

        if (freeSpace < MIN_SIZE) {
            // There isnt enough space for all the widgets, increase the size of the node
            freeSpace = MIN_SIZE;
            node.size[1] = y + widgetHeight + freeSpace * (multi.length + (!!node.imgs?.length));
            node.graph.setDirtyCanvas(true);
        }

        // Position each of the widgets
        for (const w of node.widgets) {
            w.y = y;
            if (w.type === "MONACO") {
                y += freeSpace;
                w.computedHeight = freeSpace - multi.length * 4;
            } else if (w.computeSize) {
                y += w.computeSize()[1] + 4;
            } else {
                y += LiteGraph.NODE_WIDGET_HEIGHT + 4;
            }
        }

        node.inputHeight = freeSpace;
    }

    const editorDiv = document.createElement('div');
    // Set divID to random suffix
    editorDiv.id = `monaco-editor-${Math.random().toString(36).substring(7)}`;
    document.body.append(editorDiv);
    const editor = monaco.editor.create(editorDiv, {
        automaticLayout: true,
        value: 'The embedding:test123 is a sum(cat|neg(bird)|norm(dog))',
        language: 'prompt-lang',
        theme: 'plang',
        wordBasedSuggestions: false,
        quickSuggestions: {
            strings: false,
        }
    })

    const onResizeOrig = node.onResize;
    node.onResize = function (size) {
        computeSize(size);
        if (onResizeOrig) {
            onResizeOrig.apply(this, arguments);
        }
    }

    const origOnRemoved = node.onRemoved;
    node.onRemoved = function () {
        for (const y in this.widgets) {
            if (this.widgets[y].type === 'MONACO') {
                this.widgets[y].onRemoved();
            }
        }
    }

    // TS Workaround as widgetTypes is don't know about MONACO
    const widgetType = 'MONACO' as widgetTypes;
    const widget = {
        type: widgetType,
        name: 'MONACO',
        inputEl: editorDiv,
        value: editor.getValue(),
        parent: node,
        draw: function (ctx, _, widgetWidth, y, widgetHeight) {
            if (!this?.parent?.inputHeight) {
                // If we are initially offscreen when created we wont have received a resize event
                // Calculate it here instead
                computeSize(node.size);
            }

            const margin = 10;
            const elRect = ctx.canvas.getBoundingClientRect();
            const transform = new DOMMatrix()
                .scaleSelf(elRect.width / ctx.canvas.width, elRect.height / ctx.canvas.height)
                .multiplySelf(ctx.getTransform())
                .translateSelf(margin, margin + y);

            const scale = new DOMMatrix().scaleSelf(transform.a, transform.d)
            Object.assign(this.inputEl.style, {
                transformOrigin: "0 0",
                transform: scale,
                left: `${transform.a + transform.e}px`,
                top: `${transform.d + transform.f}px`,
                width: `${widgetWidth - (margin * 2)}px`,
                height: `${this.parent.inputHeight - (margin * 2)}px`,
                // height: `400px`,
                position: "absolute",
                // background: (!node.color) ? '' : node.color,
                // color: (!node.color) ? '' : 'white',
                zIndex: app.graph._nodes.indexOf(node),
            });
        },
        onRemoved: function () {
            editor.dispose();
            widget.inputEl?.remove();
        }
    }
    return {
        widget: node.addCustomWidget(widget)
    }
}
