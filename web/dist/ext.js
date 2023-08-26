import { app } from "/scripts/app.js";
app.registerExtension({
    name: "ComfyLiterals.OtherNode",
    nodeCreated(node, app) {
        console.log("Hello from TS");
    }
});
//# sourceMappingURL=ext.js.map