import js from '@eslint/js'
import globals from 'globals'
import tseslint from 'typescript-eslint'

export default tseslint.config(
  { ignores: ['dist/**', 'node_modules/**', 'vendor/**'] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: 'module',
      globals: { ...globals.browser, ...globals.node },
      parserOptions: { ecmaFeatures: { jsx: true } },
    },
    rules: {
      // The page components legitimately return `null` while data is loading,
      // and narrow optionals by construction, so these two are off.
      '@typescript-eslint/no-explicit-any': 'error',
      'no-undef': 'off', // TypeScript already checks this, and it false-positives on types.
    },
  },
  {
    files: ['scripts/**/*.mjs'],
    languageOptions: { globals: { ...globals.node } },
  },
  {
    // Python is a different language; ESLint has no business parsing it. The
    // equivalent gate for tools/ is tools/verify_data.py, which is the real
    // check on the data pipeline.
    files: ['tools/**/*.{py,csv,tex,pdf}'],
    ignores: ['tools/**'],
    languageOptions: { globals: {} },
  },
)