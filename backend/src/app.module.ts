import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { AppController } from './app.controller.js';
import { AppService } from './app.service.js';
import { AudioJobsModule } from './audio-jobs/audio-jobs.module.js';
import configuration from './config/configuration.js';
import { MlClientModule } from './ml-client/ml-client.module.js';
import { ModelsModule } from './models/models.module.js';
import { TtsModule } from './tts/tts.module.js';

@Module({
  imports: [
    ConfigModule.forRoot({ isGlobal: true, load: [configuration] }),
    MlClientModule,
    AudioJobsModule,
    ModelsModule,
    TtsModule,
  ],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
