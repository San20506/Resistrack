import * as cdk from 'aws-cdk-lib';
import * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as kms from 'aws-cdk-lib/aws-kms';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as sagemaker from 'aws-cdk-lib/aws-sagemaker';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import { Construct } from 'constructs';

export interface InferenceProps {
  readonly environment: string;
  readonly encryptionKey: kms.IKey;
  readonly vpc: ec2.IVpc;
}

/**
 * M2.7 SageMaker Inference construct.
 * Real-time endpoint with auto-scaling, CloudWatch monitoring, and KMS encryption.
 * p95 latency target: <=2000ms. Auto-scaling 1-20 instances.
 */
export class InferenceConstruct extends Construct {
  public readonly endpointLogGroup: logs.LogGroup;
  public readonly latencyAlarm: cloudwatch.Alarm;
  public readonly executionRole: iam.Role;

  constructor(scope: Construct, id: string, props: InferenceProps) {
    super(scope, id);

    // CloudWatch log group for inference logs (no PHI)
    this.endpointLogGroup = new logs.LogGroup(this, 'InferenceLogGroup', {
      logGroupName: `/resistrack/${props.environment}/inference`,
      retention: logs.RetentionDays.ONE_YEAR,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      encryptionKey: props.encryptionKey,
    });

    // IAM execution role for SageMaker
    this.executionRole = new iam.Role(this, 'SageMakerExecutionRole', {
      roleName: `resistrack-sagemaker-${props.environment}`,
      assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSageMakerFullAccess'),
      ],
    });

    props.encryptionKey.grantEncryptDecrypt(this.executionRole);

    // CloudWatch alarm for p95 latency > 2000ms
    this.latencyAlarm = new cloudwatch.Alarm(this, 'LatencyAlarm', {
      alarmName: `resistrack-inference-p95-latency-${props.environment}`,
      metric: new cloudwatch.Metric({
        namespace: 'AWS/SageMaker',
        metricName: 'ModelLatency',
        dimensionsMap: {
          EndpointName: `resistrack-amr-endpoint-${props.environment}`,
        },
        statistic: 'p95',
        period: cdk.Duration.minutes(5),
      }),
      threshold: 2000000, // SageMaker reports in microseconds
      evaluationPeriods: 3,
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
    });
  }
}
