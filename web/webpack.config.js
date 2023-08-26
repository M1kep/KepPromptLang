const path = require('path');
const HtmlWebPackPlugin = require('html-webpack-plugin');
const webpack = require('webpack');

module.exports = {
    mode: 'development',

    entry: {
        app: './src/ext.ts',
        // 'editor.worker': 'monaco-editor/esm/vs/editor/editor.worker.js',
        // 'json.worker': 'monaco-editor/esm/vs/language/json/json.worker',
        // 'css.worker': 'monaco-editor/esm/vs/language/css/css.worker',
        // 'html.worker': 'monaco-editor/esm/vs/language/html/html.worker',
        // 'ts.worker': 'monaco-editor/esm/vs/language/typescript/ts.worker'
    },
    // externals: {
    //   "../../../../web/scripts/app.js": "import",
    //   // "../../../../web/types/comfy": "ComfyExtension"
    // },
    // externals: {
    //   '/scripts/app.js': {
    //     commonjs: '/scripts/app.js',
    //     commonjs2: '/scripts/app.js',
    //     amd: '/scripts/app.js',
    //     root: 'app'
    //   }
    // },
    resolve: {
        extensions: ['.ts', '.js'],
        // alias: {
        //   'AppAlias': '../../../../web/scripts/app.js',
        // }
    },
    output: {
        environment: {
            dynamicImport: true,
        },
        globalObject: 'self',
        filename: '[name].bundle.js',
        path: path.resolve(__dirname, 'dist')
    },
    optimization: {
        minimize: false
    },
    module: {
        rules: [
            {
                test: /\.ts?$/,
                use: 'ts-loader',
                exclude: /node_modules/
            },
            {
                test: /\.css$/,
                use: ['style-loader', 'css-loader']
            },
            {
                test: /\.ttf$/,
                use: ['file-loader']
            }
        ]
    },
    // plugins: [
    //     new HtmlWebPackPlugin({
    //         title: 'Monaco Editor Sample'
    //     }),
    //     new webpack.IgnorePlugin({
    //         resourceRegExp: /^\/scripts\/app\.js$/
    //     })
    // ]
}
