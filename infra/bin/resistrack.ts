#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { ResisTrackStack } from '../lib/resistrack-stack';

const app = new cdk.App();

new ResisTrackStack(app, 'ResisTrackStack', {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION ?? 'us-east-1',
  },
  description: 'ResisTrack - AI-Powered AMR Risk Prediction Platform',
});
