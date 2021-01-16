// For a detailed explanation regarding each configuration property, visit:
// https://jestjs.io/docs/en/configuration.html

module.exports = {
    "globals": {
        "ts-jest": {
            "tsConfig": "tsconfig.json"
        }
    },
    "moduleFileExtensions": ["ts", "js"],
    "setupFiles": [
        // "<rootDir>/test-setup.js"
    ],
    "setupFilesAfterEnv": [
        // "<rootDir>/test-setup-after-env.js"
    ],
    // testEnvironment: "node",

    "testMatch": [
        "**/*_test\\.ts"
    ],

    // An array of regexp pattern strings that are matched against all test paths, matched tests are skipped
    // testPathIgnorePatterns: [
    //   "/node_modules/"
    // ],

    "transform": {
        "^.+\\.ts$": "ts-jest"
    },

    // An array of regexp pattern strings that are matched against all source file paths, matched files will skip transformation
    // transformIgnorePatterns: [
    //     "/node_modules/"
    // ],

    // An array of regexp pattern strings that are matched against all modules before the module loader will automatically return a mock for them
    // unmockedModulePathPatterns: undefined,

    // Indicates whether each individual test should be reported during the run
    // verbose: null,

    // An array of regexp patterns that are matched against all source file paths before re-running tests in watch mode
    // watchPathIgnorePatterns: [],

    // Whether to use watchman for file crawling
    // watchman: true,
};
