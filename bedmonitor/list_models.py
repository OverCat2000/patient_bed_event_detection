import os

from openai import OpenAI

from bedmonitor import config


def main():
    config.load_dotenv(config.ROOT / ".env")
    client = OpenAI(api_key=os.environ["AWS_BEARER_TOKEN_BEDROCK"], base_url=os.environ["OPENAI_BASE_URL"],
                    project=os.getenv("OPENAI_PROJECT_ID") or None)
    for m in client.models.list():
        print(m.id)


if __name__ == "__main__":
    main()
