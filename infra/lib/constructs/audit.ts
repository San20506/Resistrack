import * as cdk from 'aws-cdk-lib';
import * as cloudtrail from 'aws-cdk-lib/aws-cloudtrail';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as kms from 'aws-cdk-lib/aws-kms';
import { Construct } from 'constructs';

export interface AuditProps {
  readonly environment: string;
  readonly encryptionKey: kms.IKey;
}

/**
 * M1.6 Audit Logging construct.
 * CloudTrail for API activity, CloudWatch Logs for audit trail.
 * HIPAA: All audit logs encrypted at rest and retained for 7 years.
 */
export class AuditConstruct extends Construct {
  public readonly trail: cloudtrail.Trail;
  public readonly auditLogGroup: logs.LogGroup;
  public readonly auditBucket: s3.Bucket;

  constructor(scope: Construct, id: string, props: AuditProps) {
    super(scope, id);

    // S3 bucket for CloudTrail logs with 7-year HIPAA retention
    this.auditBucket = new s3.Bucket(this, 'AuditLogBucket', {
      bucketName: `resistrack-audit-${props.environment}-${cdk.Aws.ACCOUNT_ID}`,
      encryption: s3.BucketEncryption.KMS,
      encryptionKey: props.encryptionKey,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      versioned: true,
      lifecycleRules: [
        {
          id: 'hipaa-retention',
          transitions: [
            {
              storageClass: s3.StorageClass.GLACIER,
              transitionAfter: cdk.Duration.days(90),
            },
          ],
          expiration: cdk.Duration.days(2555), // 7-year HIPAA retention
        },
      ],
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    // CloudWatch log group for audit trail
    this.auditLogGroup = new logs.LogGroup(this, 'AuditLogGroup', {
      logGroupName: `/resistrack/${props.environment}/audit`,
      retention: logs.RetentionDays.SEVEN_YEARS,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      encryptionKey: props.encryptionKey,
    });

    // CloudTrail for all API activity
    this.trail = new cloudtrail.Trail(this, 'AuditTrail', {
      trailName: `resistrack-audit-${props.environment}`,
      bucket: this.auditBucket,
      cloudWatchLogGroup: this.auditLogGroup,
      sendToCloudWatchLogs: true,
      encryptionKey: props.encryptionKey,
      enableFileValidation: true,
      includeGlobalServiceEvents: true,
      isMultiRegionTrail: false,
    });
  }
}
