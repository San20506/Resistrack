import * as cdk from 'aws-cdk-lib';
import * as kms from 'aws-cdk-lib/aws-kms';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import { Construct } from 'constructs';

export interface SecurityProps {
  readonly environment: string;
}

/**
 * Security construct: KMS CMKs for per-tenant encryption,
 * IAM roles with least-privilege, Secrets Manager for credentials.
 */
export class SecurityConstruct extends Construct {
  public readonly dataEncryptionKey: kms.IKey;
  public readonly auditEncryptionKey: kms.IKey;
  public readonly sagemakerRole: iam.IRole;
  public readonly lambdaRole: iam.IRole;
  public readonly dbSecret: secretsmanager.ISecret;

  constructor(scope: Construct, id: string, props: SecurityProps) {
    super(scope, id);

    // KMS CMK for data encryption (FHIR data, model artifacts)
    this.dataEncryptionKey = new kms.Key(this, 'DataEncryptionKey', {
      alias: `resistrack-${props.environment}-data`,
      description: 'CMK for ResisTrack data encryption (FHIR, models)',
      enableKeyRotation: true,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    // KMS CMK for audit log encryption
    this.auditEncryptionKey = new kms.Key(this, 'AuditEncryptionKey', {
      alias: `resistrack-${props.environment}-audit`,
      description: 'CMK for ResisTrack audit log encryption',
      enableKeyRotation: true,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    // SageMaker execution role - least privilege
    this.sagemakerRole = new iam.Role(this, 'SageMakerExecutionRole', {
      assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com'),
      description: 'SageMaker execution role for ResisTrack ML models',
      managedPolicies: [],
    });

    // Grant SageMaker role access to data encryption key
    this.dataEncryptionKey.grantEncryptDecrypt(this.sagemakerRole);

    // Lambda execution role - least privilege
    this.lambdaRole = new iam.Role(this, 'LambdaExecutionRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      description: 'Lambda execution role for ResisTrack functions',
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          'service-role/AWSLambdaVPCAccessExecutionRole'
        ),
      ],
    });

    // RDS database credentials in Secrets Manager
    this.dbSecret = new secretsmanager.Secret(this, 'RdsCredentials', {
      secretName: `resistrack/${props.environment}/rds-credentials`,
      description: 'RDS PostgreSQL credentials for ResisTrack audit database',
      generateSecretString: {
        secretStringTemplate: JSON.stringify({ username: 'resistrack_admin' }),
        generateStringKey: 'password',
        excludePunctuation: true,
        passwordLength: 32,
      },
      encryptionKey: this.auditEncryptionKey,
    });

    // Grant Lambda role access to read secrets
    this.dbSecret.grantRead(this.lambdaRole);

    // Tags
    cdk.Tags.of(this).add('Project', 'ResisTrack');
    cdk.Tags.of(this).add('Environment', props.environment);
    cdk.Tags.of(this).add('HIPAA', 'true');
  }
}
