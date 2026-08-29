import { Module } from '@nestjs/common';
import { MlClientModule } from '../ml-client/ml-client.module.js';
import { TtsController } from './tts.controller.js';

@Module({
  imports: [MlClientModule],
  controllers: [TtsController],
})
export class TtsModule {}
