# CI/CD Pipeline to deploy to AWS Lambda

1. This repo contains the source code

2. Once you commit any changes it will kick off "workflows" action which can be viewed under "https://github.com/datamatrixai/lamda-cicd-demo/actions"

3. Go to AWS->LaLambda->Functions->pythonGithubCICDFunction

4. You will see the updated changes in the Code section : 
"https://ap-south-1.console.aws.amazon.com/lambda/home?region=ap-south-1#/functions/pythonGithubCICDFunction?tab=code"

5a. You can kick off the Test event by clicking "Test" button :
https://ap-south-1.console.aws.amazon.com/lambda/home?region=ap-south-1#/functions/pythonGithubCICDFunction?tab=testing

5b. We can create a Trigger which will kick off an trigger any changes or schedule time

6. Go to Cloudwatch to see the logs CloudWatch->Log groups->/aws/lambda/pythonGithubCICDFunction)
