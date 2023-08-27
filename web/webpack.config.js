const path = require('path');
const HtmlWebPackPlugin = require('html-webpack-plugin');
const webpack = require('webpack');

module.exports = {
    mode: 'production',
    watch: true,
    entry: {
        app: './src/ext.ts',
        'editor.worker': 'monaco-editor/esm/vs/editor/editor.worker.js',
        // 'json.worker': 'monaco-editor/esm/vs/language/json/json.worker',
        // 'css.worker': 'monaco-editor/esm/vs/language/css/css.worker',
        // 'html.worker': 'monaco-editor/esm/vs/language/html/html.worker',
        // 'ts.worker': 'monaco-editor/esm/vs/language/typescript/ts.worker'
    },
    resolve: {
        extensions: ['.ts', '.js'],
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
                // loader: 'ts-loader',
                use: [{
                    loader: 'ts-loader',
                    options: {
                        transpileOnly: true,
                    }
                }],
                exclude: /node_modules/,
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
    devtool: 'eval-source-map',
    // plugins: [
    //     new HtmlWebPackPlugin({
    //         title: 'Monaco Editor Sample'
    //     }),
    //     new webpack.IgnorePlugin({
    //         resourceRegExp: /^\/scripts\/app\.js$/
    //     })
    // ]
}
