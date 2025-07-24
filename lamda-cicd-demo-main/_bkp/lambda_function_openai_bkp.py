import json
import os
import logging
from openai import OpenAI

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {
            "statusCode": 500,
            "body": json.dumps(
                {"error": "OPENAI_API_KEY environment variable not set"}
            ),
        }

    # Initialize OpenAI client with the API key
    client = OpenAI(api_key=api_key)

    user_message = event.get("message", "What is the capital of India?")

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",  # Cheapest and fastest OpenAI model
            messages=[{"role": "user", "content": user_message}],
        )
        logger.info(f"User message: {user_message}")
        logger.info(f"User message : before answer")

        answer = response.choices[0].message.content.strip()
        logger.info(f"Answer: {answer}")

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"response": answer}),
        }

    except Exception as e:
        logger.error(f"OpenAI API error: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
        }
