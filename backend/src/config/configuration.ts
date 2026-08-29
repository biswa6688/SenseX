import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const backendRoot = resolve(fileURLToPath(new URL('.', import.meta.url)), '..', '..');
const repoRoot = resolve(backendRoot, '..');

export default () => ({
  port: parseInt(process.env.PORT ?? '3001', 10),
  mlServiceUrl: process.env.ML_SERVICE_URL ?? 'http://127.0.0.1:8000',
  storageDir: process.env.STORAGE_DIR ?? resolve(repoRoot, 'storage'),
});
