import * as cdk from 'aws-cdk-lib';
import { Template, Match } from 'aws-cdk-lib/assertions';
import { ResisTrackStack } from '../lib/resistrack-stack';

describe('ResisTrackStack', () => {
  let template: Template;

  beforeAll(() => {
    const app = new cdk.App();
    const stack = new ResisTrackStack(app, 'TestStack');
    template = Template.fromStack(stack);
  });

  test('Stack synthesizes without errors', () => {
    expect(template.toJSON()).toBeDefined();
  });

  test('Creates VPC with private subnets', () => {
    template.resourceCountIs('AWS::EC2::VPC', 1);
    // HIPAA: No public subnets with internet gateways
    template.resourceCountIs('AWS::EC2::InternetGateway', 0);
  });

  test('Creates KMS keys for encryption', () => {
    template.hasResourceProperties('AWS::KMS::Key', {
      EnableKeyRotation: true,
    });
  });

  test('Creates Secrets Manager secret', () => {
    template.hasResourceProperties('AWS::SecretsManager::Secret', {
      GenerateSecretString: Match.objectLike({}),
    });
  });

  test('Creates IAM roles with descriptions', () => {
    template.hasResourceProperties('AWS::IAM::Role', {
      Description: Match.anyValue(),
    });
  });
});
