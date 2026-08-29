import { Module } from '@nestjs/common';
import { MlClientModule } from '../ml-client/ml-client.module.js';
import { StorageModule } from '../storage/storage.module.js';
import { AudioJobsController } from './audio-jobs.controller.js';
import { AudioJobsService } from './audio-jobs.service.js';

@Module({
  imports: [MlClientModule, StorageModule],
  controllers: [AudioJobsController],
  providers: [AudioJobsService],
})
export class AudioJobsModule {}
