build-image:
	docker build -t stockpred -f Dockerfile .

run:
	mlflow run .

# https://mlflow.org/docs/latest/models.html#example-using-the-custom-sktime-flavor
serve:
	mlflow models serve -m runs:/8f4a735b0ab94d63aa6371ff794ab5a6/model_random_forest  --env-manager conda --host 127.0.0.1 --port 5001
# mlflow models serve -m runs:/my-run-id/model-path &

test-endpoint:
	curl http://127.0.0.1:5001/invocations -H 'Content-Type:application/json' -d '{
		"dataframe_split": {
			"data": [[0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0], [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1]]
			}
		}'
