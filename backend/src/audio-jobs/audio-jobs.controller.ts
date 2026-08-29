import {
  Controller,
  Get,
  Param,
  Post,
  Res,
  UploadedFile,
  UseInterceptors,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { ApiConsumes, ApiOperation, ApiTags } from '@nestjs/swagger';
import type { Response } from 'express';
import { AudioJobsService } from './audio-jobs.service.js';

@ApiTags('audio-jobs')
@Controller('api/audio-jobs')
export class AudioJobsController {
  constructor(private readonly audioJobs: AudioJobsService) {}

  @Post()
  @ApiConsumes('multipart/form-data')
  @ApiOperation({ summary: 'Upload audio and start the STT/diarization/summary/sentiment/QA pipeline' })
  @UseInterceptors(FileInterceptor('file'))
  async create(@UploadedFile() file: Express.Multer.File) {
    return this.audioJobs.submit(file.originalname, file.buffer);
  }

  @Get()
  @ApiOperation({ summary: 'List past jobs (from shared storage, no separate DB)' })
  history() {
    return this.audioJobs.history();
  }

  @Get(':id')
  status(@Param('id') id: string) {
    return this.audioJobs.status(id);
  }

  @Get(':id/result')
  result(@Param('id') id: string) {
    return this.audioJobs.result(id);
  }

  @Get(':id/audio/original')
  async streamOriginal(@Param('id') id: string, @Res() res: Response) {
    const path = this.audioJobs.originalAudioPath(id);
    this.audioJobs.streamFile(path).pipe(res);
  }

  @Get(':id/audio/summary')
  async streamSummary(@Param('id') id: string, @Res() res: Response) {
    const path = this.audioJobs.summaryAudioPath(id);
    res.setHeader('Content-Type', 'audio/wav');
    this.audioJobs.streamFile(path).pipe(res);
  }
}
