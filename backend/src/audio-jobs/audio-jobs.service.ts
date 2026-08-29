import { Injectable, NotFoundException } from '@nestjs/common';
import { createReadStream, existsSync } from 'node:fs';
import { MlClientService, MlJobStatus } from '../ml-client/ml-client.service.js';
import { StorageService } from '../storage/storage.service.js';

@Injectable()
export class AudioJobsService {
  constructor(
    private readonly mlClient: MlClientService,
    private readonly storage: StorageService,
  ) {}

  async submit(originalName: string, buffer: Buffer): Promise<{ jobId: string; status: string }> {
    const filePath = this.storage.saveUpload(originalName, buffer);
    return this.mlClient.createJob(filePath);
  }

  async status(jobId: string): Promise<MlJobStatus> {
    return this.mlClient.getJobStatus(jobId);
  }

  async result(jobId: string): Promise<Record<string, unknown>> {
    return this.mlClient.getJobResult(jobId);
  }

  /** History via the shared filesystem — ml-service's job.json sidecar is
   * the single source of truth, no separate backend DB. */
  history(): Array<Record<string, unknown>> {
    return this.storage
      .listJobIds()
      .map((id) => this.storage.readJobSidecar(id))
      .filter((sidecar): sidecar is Record<string, unknown> => sidecar !== null)
      .sort((a, b) => (b.createdAt as number) - (a.createdAt as number));
  }

  originalAudioPath(jobId: string): string {
    const sidecar = this.storage.readJobSidecar(jobId);
    if (!sidecar) throw new NotFoundException('job not found');
    const dir = this.storage.jobDir(jobId);
    for (const ext of ['.wav', '.mp3', '.m4a', '.webm', '.ogg']) {
      const candidate = `${dir}/original${ext}`;
      if (existsSync(candidate)) return candidate;
    }
    throw new NotFoundException('original audio not found');
  }

  summaryAudioPath(jobId: string): string {
    const path = `${this.storage.jobDir(jobId)}/summary.wav`;
    if (!existsSync(path)) throw new NotFoundException('summary audio not found');
    return path;
  }

  streamFile(path: string) {
    return createReadStream(path);
  }
}
