import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as kms from 'aws-cdk-lib/aws-kms';
import * as rds from 'aws-cdk-lib/aws-rds';
import * as s3 from 'aws-cdk-lib/aws-s3';
import { Construct } from 'constructs';

export interface StorageProps {
  readonly vpc: ec2.IVpc;
  readonly encryptionKey: kms.IKey;
  readonly environment: string;
}

/**
 * M1.5 Data Storage Layer — RDS PostgreSQL (audit), S3 (data lake), lifecycle policies.
 * All storage encrypted at rest with KMS CMK. VPC-only access. 7-year retention.
 */
export class StorageConstruct extends Construct {
  public readonly auditDatabase: rds.DatabaseInstance;
  public readonly dataLakeBucket: s3.Bucket;
  public readonly modelArtifactsBucket: s3.Bucket;

  constructor(scope: Construct, id: string, props: StorageProps) {
    super(scope, id);

    // RDS PostgreSQL for audit logs and clinical event tracking
    const dbSecurityGroup = new ec2.SecurityGroup(this, 'AuditDbSg', {
      vpc: props.vpc,
      description: 'Security group for ResisTrack audit RDS instance',
      allowAllOutbound: false,
    });

    this.auditDatabase = new rds.DatabaseInstance(this, 'AuditDatabase', {
      engine: rds.DatabaseInstanceEngine.postgres({
        version: rds.PostgresEngineVersion.VER_15,
      }),
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MEDIUM),
      vpc: props.vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_ISOLATED },
      securityGroups: [dbSecurityGroup],
      databaseName: 'resistrack_audit',
      credentials: rds.Credentials.fromGeneratedSecret('resistrack_admin', {
        secretName: `resistrack/${props.environment}/audit-db-credentials`,
        encryptionKey: props.encryptionKey,
      }),
      storageEncrypted: true,
      storageEncryptionKey: props.encryptionKey,
      multiAz: props.environment === 'prod',
      backupRetention: cdk.Duration.days(35),
      deletionProtection: props.environment === 'prod',
      removalPolicy: props.environment === 'prod'
        ? cdk.RemovalPolicy.RETAIN
        : cdk.RemovalPolicy.DESTROY,
      enablePerformanceInsights: true,
      performanceInsightEncryptionKey: props.encryptionKey,
      cloudwatchLogsExports: ['postgresql'],
    });

    // S3 Data Lake bucket with lifecycle policies
    this.dataLakeBucket = new s3.Bucket(this, 'DataLakeBucket', {
      bucketName: `resistrack-data-lake-${props.environment}`,
      encryption: s3.BucketEncryption.KMS,
      encryptionKey: props.encryptionKey,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      versioned: true,
      enforceSSL: true,
      removalPolicy: props.environment === 'prod'
        ? cdk.RemovalPolicy.RETAIN
        : cdk.RemovalPolicy.DESTROY,
      lifecycleRules: [
        {
          id: 'TransitionToIA',
          transitions: [
            {
              storageClass: s3.StorageClass.INFREQUENT_ACCESS,
              transitionAfter: cdk.Duration.days(90),
            },
            {
              storageClass: s3.StorageClass.GLACIER,
              transitionAfter: cdk.Duration.days(365),
            },
          ],
          // 7-year HIPAA retention
          expiration: cdk.Duration.days(2555),
        },
      ],
    });

    // S3 bucket for ML model artifacts
    this.modelArtifactsBucket = new s3.Bucket(this, 'ModelArtifactsBucket', {
      bucketName: `resistrack-model-artifacts-${props.environment}`,
      encryption: s3.BucketEncryption.KMS,
      encryptionKey: props.encryptionKey,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      versioned: true,
      enforceSSL: true,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });
  }
}
