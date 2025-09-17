from edgeimpulse_api import IngestionApi

api = IngestionApi(api_key="ei_...")
response = api.upload_sample(
    category="training",
    label="bird",
    file_path="./dataset/images/09754.jpg"
)
print(response)