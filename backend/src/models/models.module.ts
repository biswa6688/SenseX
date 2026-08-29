import { Module } from '@nestjs/common';
import { MlClientModule } from '../ml-client/ml-client.module.js';
import { ModelsController } from './models.controller.js';

@Module({
  imports: [MlClientModule],
  controllers: [ModelsController],
})
export class ModelsModule {}
