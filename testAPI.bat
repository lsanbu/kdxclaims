curl -X POST "http://127.0.0.1:8000/upload-claim" ^
  -H "Content-Type: application/json" ^
  -d "{
    \"telegram_user_id\": 7055992162,
    \"telegram_chat_id\": 7055992162,
    \"claim_type\": \"fuel\",
    \"claim_date\": \"2025-08-24\",
    \"claim_time\": \"17:18\",
    \"total_rs\": 309.76,
    \"station\": \"Bharat Petroleum\",
    \"reference_no\": \"243622025H101809\",
    \"rate_rs_per_l\": 100.90,
    \"volume_l\": 3.07,
    \"notes\": \"Test from curl\"
}"
