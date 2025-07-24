======================================================================================================
Git access to supertails_gcp_aws_transfer repository in Github =>
1. git@github.com:datamatrixai/supertails_gcp_aws_transfer.git
2. https://github.com/datamatrixai/supertails_gcp_aws_transfer.git
======================================================================================================
Step 1: to create the repository (https://github.com/datamatrixai/supertails_gcp_aws_transfer.git)
Step 2: Using Git Bash, clone git repo to a windows directory. Map the same folder to Visual Studio code
Step 3: Structure of GitHub Repo
	repo/
	├── lambda_function.py         # your handler
	├── requirements.txt           # (if needed)
	└── .github/
		└── workflows/
			└── deploy.yml         # GitHub Actions workflow
Step 4: 
	Implement lambda_function.py and requirements.txt based on your requirement
	https://github.com/datamatrixai/supertails_gcp_aws_transfer/blob/main/lambda_function.py
	https://github.com/datamatrixai/supertails_gcp_aws_transfer/blob/main/requirements.txt

Step 5: To create GitHub Actions Workflow, need to implement deploy.yml which will be ultimaely deploy in AWS lambda 
	https://github.com/datamatrixai/supertails_gcp_aws_transfer/blob/main/.github/workflows/deploy.yml

Step 6: Create a Lambda Function in AWS
	6.1 Go to the AWS Console → Lambda
	6.2 Click "Create function"
	6.3 Choose: 
			Author from scratch
			Function name: supertails_gcp_aws_transfer ✅ 
			Runtime: Python 3.11
			Role: Create a new role with basic Lambda permissions
				Click Create Function (This will create the Lambda function where we will deploy our code)

Step 7: Create an IAM User with Programmatic Access (This user will be used by GitHub Actions to deploy to AWS)
👉 Please note: I have already created this user "github-ci-deploy" for previous setup, will reuse it keys
	7.1 Go to AWS Console → IAM → Users → Add user
	7.2 Name: github-ci-deploy ✅ 
	7.3 Enable Programmatic access
	7.4 Attach policies:
		AWSLambda_FullAccess
		IAMReadOnlyAccess (for function roles)
		CloudWatchLogsFullAccess (for logging)

Step 8: Once github-ci-deploy user created, we need to generate all the secrete keys
	IAM -> Users -> github-ci-deploy (your user) -> Security credentials -> Access keys 
		a. Click on "Create access key" button to create new one
		b. Existing keys will be shown if already created
	8.1 AWS_ACCESS_KEY_ID = ""
	8.2 AWS_SECRET_ACCESS_KEY = ""
	8.3 AWS_REGION = "" (you can view it next to Setting button in the top bar)
	8.4 LAMBDA_FUNCTION_NAME = supertails_gcp_aws_transfer_function (same name as mention in Step 6.3)
	Note : The above keys are only for github-ci-deploy user, it will be different for other users

Step 9: Add AWS Credentials to GitHub Secrets
	9.1 Go to GitHub repo: https://github.com:datamatrixai/supertails_gcp_aws_transfer
	9.2 Go to Settings → Secrets and Variables → Actions → New repository secret
	9.3 Add 4 Repository secrets as mentioned in Step 8.1, 8.2, 8.3, 8.4

Step 10: Add OpenAI library to Lambda deployment package (Note: The Lambda runtime does not include the openai Python package automatically). Using Lambda Layer, we can upload the OpenAI packages for reusability
	10.1 Run the following in Git Bash one by one
		a. mkdir gcp-aws-dependencies && cd gcp-aws-dependencies
		b. echo -e "openai==1.30.1\nhttpx==0.27.0" > requirements.txt
		c. pip install -r requirements.txt -t python
		d. zip -r gcp-aws-dependencies.zip python
	10.2 In the AWS Console => Go to Lambda => Layers => Click "Create Layer"
		Name :  "gcp-aws-dependencies"
		Upload : gcp-aws-dependencies.zip (which created in Step 10.1)
		Compatible runtimes: Python 3.11
		Click Create button
	10.3 Attach this layer "gcp-aws-dependencies" to Lambda function "supertails_gcp_aws_transfer"
		a. Go to AWS Lambda > Functions
		b. Click Lambda function "supertails_gcp_aws_transfer"
		c. Go to "Layers" under Function overview
		d. Click [Add a layer]
		e. Choose: "Custom layers" or "ARN"
		f. Select "gcp-aws-dependencies"
		g. Pick the latest version (manually step, everytime we have to update for any new uploaded version
		h. Click Add

Step 11 JSON key of GCP Service Account Key
	Step 1: Create and Download Service Account Key
		a. Go to the Google Cloud Console
		b. Navigate to IAM & Admin > Service Accounts
		c. Find your service account: client-storage-access@my-python-project-461822.iam.gserviceaccount.com
		d. Click on the service account, then go to the Keys tab
		e. Click Add Key > Create New Key > JSON
		f. Download the JSON key file (keep it secure!)
	Step 2: Grant Necessary Permissions
		a. In Google Cloud Console, go to IAM & Admin > IAM
		b. Find your service account and ensure it has:
			Storage Object Admin (for full read/write access)
			OR Storage Object Viewer (for read-only access)
			OR Storage Object Creator (for write access)
	Step 3: Encode your JSON key file to base64 in Git Bash
		cd D:/PythonGithubAWSLambda/gcp-json-key/
		base64 -i my-python-project-461822-62ddee6bd82b.json > base64-key

Step 12: Set GCP_SERVICE_ACCOUNT_KEY
	12.1 Go to Lambda -> Functions -> supertails_gcp_aws_transfer
	12.2 Click Configuration
	12.3 Environment variables
	12.4 Add "GCP_SERVICE_ACCOUNT_KEY" and set base64-key content
	
Step 13: Add permissions in the role "supertails_gcp_aws_transfer_function-role-620mbgwz"
	13.1 Go to location : IAM > Roles > supertails_gcp_aws_transfer_function-role-620mbgwz
	13.2 Add Permissions > Create inline policy > add the following json to give necessary permissions in the supertails-lambda-output-bucket s3 bucket
	{
	  "Version": "2012-10-17",
	  "Statement": [
		{
		  "Sid": "S3BucketAccess",
		  "Effect": "Allow",
		  "Action": [
			"s3:ListBucket",
			"s3:GetBucketLocation"
		  ],
		  "Resource": [
			"arn:aws:s3:::supertails-lambda-output-bucket"
		  ]
		},
		{
		  "Sid": "S3ObjectAccess",
		  "Effect": "Allow",
		  "Action": [
			"s3:GetObject",
			"s3:PutObject",
			"s3:DeleteObject",
			"s3:PutObjectAcl"
		  ],
		  "Resource": [
			"arn:aws:s3:::supertails-lambda-output-bucket/*"
		  ]
		}
	  ]
	}
	13.3 In the next review screen, give name - S3AccessPolicy and save.
======================================================================================================
Open "git bash", run the commands to perform operations locally
aws s3 ls supertails-lambda-output-bucket/supertails-vet-audit --recursive
aws s3 rm s3://supertails-lambda-output-bucket/supertails-vet-audit/sample1.txt
aws s3 rm s3://supertails-lambda-output-bucket/supertails-vet-audit/ --recursive
aws s3 cp s3://supertails-lambda-output-bucket/supertails-vet-audit/blob_test.txt ./
aws s3 cp D:/Supertails/blob_test_1.txt s3://supertails-lambda-output-bucket/supertails-vet-audit/
aws s3 cp s3://supertails-lambda-output-bucket/supertails-vet-audit/blob_test.txt s3://supertails-lambda-output-bucket/supertails-vet-audit/blob_test_bkp.txt
======================================================================================================
Open "Google Could SDK Shell", run the commands to perform operations locally
gcloud auth login ==> Allow in the browser
gcloud auth application-default login ==> Allow in the browser
gcloud config set project my-python-project
gsutil ls gs://reetesh-bucket-2025-v1
gsutil ls -r gs://reetesh-bucket-2025-v1
gsutil cp gs://reetesh-bucket-2025-v1/sample1.txt D:\Supertails\
gsutil cp D:\Supertails\blob_test_1.txt gs://reetesh-bucket-2025-v1/test/
gsutil rm gs://reetesh-bucket-2025-v1/test/blob_test_1.txt
gsutil rm -r gs://reetesh-bucket-2025-v1/test/
======================================================================================================
// Test data
{
  "operation": "download_single_file_GCS_to_S3",
  "file_name": "sample1.txt"
}

// optional
{
  "operation": "download_single_file_GCS_to_S3",
  "file_name": "05-May-2025/test1.txt"
}

{
  "operation": "download_directory_GCS_to_S3",
  "directory_name": "07-May-2025"
}	

{
  "operation": "upload_single_file_S3_to_GCS",
  "file_name": "sample1.txt"
}

{
  "operation": "upload_directory_S3_to_GCS",
  "directory_name": "06-May-2025"
}

// Related GCS operations
{
  "operation": "delete_single_file_in_gcs",
  "file_name": "blob_test1.txt"
}

{
  "operation": "delete_directory_in_gcs",
  "directory_name": "06-May-2025"
}

{
  "operation": "rename_single_file_in_gcs",
  "source_file": "blob_test1.txt",
  "destination_file": "blob_test2.txt"
}

{
  "operation": "rename_directory_in_gcs",
  "source_directory": "08-May-2025",
  "destination_directory": "07-May-2025"
}

// Related S3 operations
{
	"operation": "delete_single_file_in_s3",
	"file_name": "supertails-vet-audit/05-May-2025/sample1.txt"
}
{
	"operation": "delete_directory_in_s3",
	"directory_name": "supertails-vet-audit/05-May-2025/"
}
{
	"operation": "move_single_file_in_s3",
	"source_file_name": "supertails-vet-audit/06-May-2025/sample3.txt",
	"destination_file_name": "supertails-vet-audit/07-May-2025/sample4.txt"
}
{
	"operation": "move_directory_in_s3",
	"source_directory_name": "supertails-vet-audit/06-May-2025",
	"destination_directory_name": "supertails-vet-audit/08-May-2025/"
}
{
	"operation": "copy_single_file_in_s3",
	"source_file_name": "supertails-vet-audit/07-May-2025/test2.txt",
	"destination_file_name": "supertails-vet-audit/07-May-2025/test2.txt"
}
{
	"operation": "copy_directory_in_s3",
	"source_directory_name": "supertails-vet-audit/07-May-2025",
	"destination_directory_name": "supertails-vet-audit/09-May-2025"
}
======================================================================================================