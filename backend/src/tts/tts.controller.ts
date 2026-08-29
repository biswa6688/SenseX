import { Body, Controller, Post, Res } from '@nestjs/common';
import { ApiTags } from '@nestjs/swagger';
import type { Response } from 'express';
import { createReadStream } from 'node:fs';
import { MlClientService } from '../ml-client/ml-client.service.js';

class SpeakDto {
  text!: string;
  voice?: string;
}

@ApiTags('tts')
@Controller('api/tts')
export class TtsController {
  constructor(private readonly mlClient: MlClientService) {}

  @Post()
  async speak(@Body() body: SpeakDto, @Res() res: Response) {
    const { audioPath } = await this.mlClient.speak(body.text, body.voice);
    res.setHeader('Content-Type', 'audio/wav');
    createReadStream(audioPath).pipe(res);
  }
}
