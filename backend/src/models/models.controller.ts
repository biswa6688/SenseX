import { Controller, Get, Param, Post } from '@nestjs/common';
import { ApiTags } from '@nestjs/swagger';
import { MlClientService } from '../ml-client/ml-client.service.js';

@ApiTags('models')
@Controller('api/models')
export class ModelsController {
  constructor(private readonly mlClient: MlClientService) {}

  @Get()
  list() {
    return this.mlClient.listModels();
  }

  @Post(':id/download')
  download(@Param('id') id: string) {
    return this.mlClient.downloadModel(id);
  }

  @Get(':id/status')
  status(@Param('id') id: string) {
    return this.mlClient.modelStatus(id);
  }
}
