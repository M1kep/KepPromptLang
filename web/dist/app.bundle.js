/*
 * ATTENTION: The "eval" devtool has been used (maybe by default in mode: "development").
 * This devtool is neither made for production nor for readable output files.
 * It uses "eval()" calls to create a separate source file in the browser devtools.
 * If you are trying to read the output file, select a different devtool (https://webpack.js.org/configuration/devtool/)
 * or disable the default devtool with "devtool: false".
 * If you are looking for production-ready output files, see mode: "production" (https://webpack.js.org/configuration/mode/).
 */
/******/ (() => { // webpackBootstrap
/******/ 	"use strict";
/******/ 	var __webpack_modules__ = ({

/***/ "./src/ext.ts":
/*!********************!*\
  !*** ./src/ext.ts ***!
  \********************/
/***/ (() => {

eval("\n// import {app} from /* webpackIgnore: true */\"/scripts/app.js\";\nconst app = import(/* webpackIgnore: true */ '/scripts/app.js').then(({ app }) => {\n    app.registerExtension({\n        name: \"ComfyLiterals.OtherNode\",\n        nodeCreated(node, app) {\n            // node.horizontal = true;\n            console.log(\"Hello from TS\");\n            //     const onAdded = node.onAdded\n            //     node.onAdded = function (graph) {\n            //         console.log(\"OperationNode onAdded\")\n            //         const firstCallbackResp = onAdded ? onAdded.apply(this, arguments) : undefined;\n            //\n            //         /**\n            //          * @type {Record<string, INodeInputSlot>}\n            //          */\n            //         const inputCache = {\n            //             \"A\": node.inputs[1],\n            //             \"B\": node.inputs[3]\n            //         }\n            //\n            //         if (this.widgets_values) {\n            //             const aType = this.widgets_values[0]\n            //             const bType = this.widgets_values[1]\n            //\n            //             // [IntA, FloatA, IntB, FloatB]\n            //             const aIdxToDelete = aType === \"INT\" ? 1 : 0\n            //             // [*A, IntB, FloatB]\n            //             const bIdxToDelete = bType === \"INT\" ? 3 : 1\n            //\n            //             inputCache[\"A\"] = node.inputs[aIdxToDelete]\n            //             this.removeInput(aIdxToDelete)\n            //             inputCache[\"B\"] = node.inputs[bIdxToDelete]\n            //             this.removeInput(bIdxToDelete)\n            //         } else {\n            //             // Nodes being restored/pasted don't have widget_values\n            //             // Node has 4 inputs(IntA, FloatA, IntB, FloatB)\n            //             // Remove both float inputs, Float B moves to index 2 after Float A is removed\n            //             this.removeInput(1)\n            //             this.removeInput(2)\n            //         }\n            //\n            //         // Add a toggle widget to the node\n            //         this.widgets[0].callback = function (v, canvas, node) {\n            //             addInputAtIndex(node, inputCache[\"A\"], 0)\n            //             inputCache[\"A\"] = node.inputs[1]\n            //             node.removeInput(1)\n            //         }\n            //         this.widgets[1].callback = function (v, canvas, node) {\n            //             addInputAtIndex(node, inputCache[\"B\"], 2)\n            //             inputCache[\"B\"] = node.inputs[1]\n            //             node.removeInput(1)\n            //         }\n            //     }\n            // }\n        }\n    });\n});\n\n\n//# sourceURL=webpack:///./src/ext.ts?");

/***/ })

/******/ 	});
/************************************************************************/
/******/ 	
/******/ 	// startup
/******/ 	// Load entry module and return exports
/******/ 	// This entry module can't be inlined because the eval devtool is used.
/******/ 	var __webpack_exports__ = {};
/******/ 	__webpack_modules__["./src/ext.ts"]();
/******/ 	
/******/ })()
;