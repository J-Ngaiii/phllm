from phllm.config import check_status, get_config

for model_name in get_config().keys():
    print(f"Status of '{model_name}': {check_status(model_name)}")
print(f"Model status check complete!")