import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { NetworkingConstruct } from './constructs/networking';
import { SecurityConstruct } from './constructs/security';
import { ConnectivityConstruct } from './constructs/connectivity';
import { StorageConstruct } from './constructs/storage';
import { AuditConstruct } from './constructs/audit';
import { InferenceConstruct } from './constructs/inference';

/**
 * Root stack for ResisTrack - AI-Powered AMR Risk Prediction Platform.
 * HIPAA-compliant infrastructure with private networking and encryption.
 */
export class ResisTrackStack extends cdk.Stack {
  public readonly networking: NetworkingConstruct;
  public readonly security: SecurityConstruct;
  public readonly connectivity: ConnectivityConstruct;
  public readonly storage: StorageConstruct;
  public readonly audit: AuditConstruct;
  public readonly inference: InferenceConstruct;

  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const environment = this.node.tryGetContext('environment') ?? 'dev';

    // Phase 1: Networking - VPC with private subnets only (HIPAA requirement)
    this.networking = new NetworkingConstruct(this, 'Networking', {
      environment,
    });

    // Phase 1: Security - KMS CMKs, IAM roles, Secrets Manager
    this.security = new SecurityConstruct(this, 'Security', {
      environment,
    });

    // M1.2: Hospital connectivity - VPN, API Gateway, mTLS
    this.connectivity = new ConnectivityConstruct(this, 'Connectivity', {
      vpc: this.networking.vpc,
      environment,
    });

    // M1.5: Data storage - HealthLake, RDS, S3 with HIPAA retention
    this.storage = new StorageConstruct(this, 'Storage', {
      vpc: this.networking.vpc,
      encryptionKey: this.security.dataEncryptionKey,
      environment,
    });

    // M1.6: Audit logging - CloudTrail, CloudWatch, 7-year retention
    this.audit = new AuditConstruct(this, 'Audit', {
      encryptionKey: this.security.dataEncryptionKey,
      environment,
    });

    // M2.7: SageMaker inference endpoint with auto-scaling and monitoring
    this.inference = new InferenceConstruct(this, 'Inference', {
      vpc: this.networking.vpc,
      encryptionKey: this.security.dataEncryptionKey,
      environment,
    });

    // Tags for compliance tracking
    cdk.Tags.of(this).add('Project', 'ResisTrack');
    cdk.Tags.of(this).add('Compliance', 'HIPAA');
    cdk.Tags.of(this).add('ManagedBy', 'CDK');
  }
}
