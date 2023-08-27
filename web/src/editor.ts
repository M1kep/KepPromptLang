import * as monaco from 'monaco-editor';

export function initEditor() {
    monaco.languages.register({id: 'prompt-lang'});
    monaco.languages.setMonarchTokensProvider('prompt-lang', {
        defaultToken: 'invalid',
        tokenizer: {
            root: [
                [/(embedding:)([A-Za-z0-9,_-]+)/, ['keyword', 'embedding']],
                [/(")[^"\\]*(\\.[^"\\]*)*(")|(')[^'\\]*(\\.[^'\\]*)*(')/, ['string']],
                [/(sum|neg|norm|diff)\(/, 'function'],
                [/[A-Za-z0-9,_-]+/, 'word']
            ]
        }
    });
    monaco.languages.setLanguageConfiguration('prompt-lang', {
        brackets: [
            ['(', ')']
        ],
        autoClosingPairs: [
            {open: '(', close: ')'},
            {open: '"', close: '"', notIn: ['string']},
            {open: "'", close: "'", notIn: ['string']}
        ],
        surroundingPairs: [
            ['(', ')'],
            ['"', '"'],
            ["'", "'"]
        ],
        indentationRules: {
            // You can adjust the increase/decrease indent patterns as needed
            increaseIndentPattern: /([({\[])$/,
            decreaseIndentPattern: /^([)}\]])/
        }
    });
    monaco.editor.defineTheme('plang', {
        base: 'vs',
        inherit: true,
        colors: {},
        rules: [
            // { token: 'embedding', foreground: '009688' },
            {token: 'embedding.prompt-lang', foreground: '00FF00'},
            // { token: 'function', foreground: 'FF5722', fontStyle: 'bold' },
            {token: 'function', foreground: 'FF0000', fontStyle: 'bold'},
            // { token: 'string', foreground: '3F51B5' },
            {token: 'string', foreground: '0000FF'},
            // { token: 'word', foreground: '795548' }
            {token: 'word', foreground: 'FF00FF'}
        ]
    });
}
