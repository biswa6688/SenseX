import { HttpModule } from '@nestjs/axios';
import { Module } from '@nestjs/common';
import { MlClientService } from './ml-client.service.js';

@Module({
  imports: [HttpModule.register({ timeout: 15_000 })],
  providers: [MlClientService],
  exports: [MlClientService],
})
export class MlClientModule {}
