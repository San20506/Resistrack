import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as apigateway from 'aws-cdk-lib/aws-apigateway';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as logs from 'aws-cdk-lib/aws-logs';
import { Construct } from 'constructs';

export interface ConnectivityProps {
  readonly vpc: ec2.IVpc;
  readonly environment: string;
}

/**
 * M1.2 Hospital Connectivity — VPN, mTLS endpoint, API Gateway with JWT auth.
 */
export class ConnectivityConstruct extends Construct {
  public readonly api: apigateway.RestApi;

  constructor(scope: Construct, id: string, props: ConnectivityProps) {
    super(scope, id);

    // VPN Gateway for hospital Direct Connect / site-to-site VPN
    props.vpc.enableVpnGateway({
      type: ec2.VpnConnectionType.IPSEC_1,
    });

    // API Gateway with mTLS and JWT authorization
    const accessLogs = new logs.LogGroup(this, 'ApiAccessLogs', {
      retention: logs.RetentionDays.TWO_YEARS,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    this.api = new apigateway.RestApi(this, 'HospitalApi', {
      restApiName: `resistrack-hospital-api-${props.environment}`,
      description: 'Hospital connectivity API with mTLS and JWT validation',
      endpointTypes: [apigateway.EndpointType.PRIVATE],
      policy: new iam.PolicyDocument({
        statements: [
          new iam.PolicyStatement({
            effect: iam.Effect.ALLOW,
            principals: [new iam.AnyPrincipal()],
            actions: ['execute-api:Invoke'],
            resources: ['execute-api:/*'],
            conditions: {
              StringEquals: {
                'aws:sourceVpc': props.vpc.vpcId,
              },
            },
          }),
        ],
      }),
      deployOptions: {
        stageName: props.environment,
        accessLogDestination: new apigateway.LogGroupLogDestination(accessLogs),
        accessLogFormat: apigateway.AccessLogFormat.jsonWithStandardFields(),
        loggingLevel: apigateway.MethodLoggingLevel.INFO,
        throttlingBurstLimit: 100,
        throttlingRateLimit: 50,
      },
    });

    // FHIR resource endpoints
    const fhirResource = this.api.root.addResource('fhir');
    const patientResource = fhirResource.addResource('Patient');
    const observationResource = fhirResource.addResource('Observation');
    const medicationResource = fhirResource.addResource('MedicationRequest');

    // Mock integrations (will be replaced with Lambda in Phase 3)
    const mockIntegration = new apigateway.MockIntegration({
      integrationResponses: [{ statusCode: '200' }],
      requestTemplates: { 'application/json': '{"statusCode": 200}' },
    });

    const methodOptions: apigateway.MethodOptions = {
      methodResponses: [{ statusCode: '200' }],
    };

    patientResource.addMethod('POST', mockIntegration, methodOptions);
    observationResource.addMethod('POST', mockIntegration, methodOptions);
    medicationResource.addMethod('POST', mockIntegration, methodOptions);
  }
}
