import { HttpService } from '@nestjs/axios';
import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { firstValueFrom } from 'rxjs';

export interface MlJobStatus {
  jobId: string;
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
  stage: string | null;
  error: string | null;
  stages: string[];
  elapsedSeconds: number | null;
  etaSeconds: number | null;
  queuePosition: number | null;
  queueLength: number | null;
}

export interface MlModelInfo {
  id: string;
  name: string;
  repo: string;
  requiresAuth: boolean;
  authNote?: string;
  downloaded: boolean;
}

@Injectable()
export class MlClientService {
  private readonly baseUrl: string;

  constructor(
    private readonly http: HttpService,
    private readonly config: ConfigService,
  ) {
    this.baseUrl = this.config.get<string>('mlServiceUrl')!;
  }

  async createJob(filePath: string): Promise<{ jobId: string; status: string }> {
    const { data } = await firstValueFrom(
      this.http.post(`${this.baseUrl}/jobs`, null, { params: { file_path: filePath } }),
    );
    return data;
  }

  async getJobStatus(jobId: string): Promise<MlJobStatus> {
    const { data } = await firstValueFrom(this.http.get(`${this.baseUrl}/jobs/${jobId}`));
    return data;
  }

  /** Raw SSE byte stream from ml-service — piped straight through to the
   * frontend response rather than re-parsed/re-emitted, so the backend
   * stays a dumb proxy for this one (see audio-jobs.controller.ts). */
  async streamJobStatus(jobId: string) {
    const { data } = await firstValueFrom(
      this.http.get(`${this.baseUrl}/jobs/${jobId}/stream`, { responseType: 'stream' }),
    );
    return data as NodeJS.ReadableStream;
  }

  async getJobResult(jobId: string): Promise<Record<string, unknown>> {
    const { data } = await firstValueFrom(this.http.get(`${this.baseUrl}/jobs/${jobId}/result`));
    return data;
  }

  async cancelJob(jobId: string): Promise<MlJobStatus> {
    const { data } = await firstValueFrom(this.http.post(`${this.baseUrl}/jobs/${jobId}/cancel`));
    return data;
  }

  async speak(text: string, voice?: string): Promise<{ audioPath: string }> {
    const { data } = await firstValueFrom(
      this.http.post(`${this.baseUrl}/tts`, { text, voice }),
    );
    return data;
  }

  async listModels(): Promise<MlModelInfo[]> {
    const { data } = await firstValueFrom(this.http.get(`${this.baseUrl}/models`));
    return data;
  }

  async downloadModel(modelId: string): Promise<MlModelInfo> {
    const { data } = await firstValueFrom(
      this.http.post(`${this.baseUrl}/models/${modelId}/download`),
    );
    return data;
  }

  async modelStatus(modelId: string): Promise<{ id: string; downloaded: boolean }> {
    const { data } = await firstValueFrom(
      this.http.get(`${this.baseUrl}/models/${modelId}/status`),
    );
    return data;
  }
}
