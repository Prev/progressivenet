const path = require('path');

module.exports = {
    entry: {
        ssd: './src/ssd.ts',
        classification: './src/classification.ts',
    },
    output: {
        filename: '[name].js',
        path: path.join(__dirname, 'dist'),
    },
    resolve: {
        extensions: ['.ts', '.js', '.json'],
    },
    node: {
        fs: 'empty'
    },
    module: {
        rules: [
            {
                test: /\.ts$/,
                loader: [
                    {
                        loader: 'ts-loader'
                    },
                ],
                exclude: /node_modules/,
            },
        ]
	},
	devServer: {
        static: path.join(__dirname, 'dist'),
        port: 8000,
    }
};
