import base64
import httpx
from fastapi import APIRouter, HTTPException, UploadFile, File

router = APIRouter()

# Microservices endpoints
MICROSERVICES = {
    "cin_detection": "http://cin-detection:8001/detect-cin/",
    "field_extraction": "http://field-extraction:8002/extract-fields/",
    "preprocessing": "http://preprocessing:8003/preprocess/",
    "ocr": "http://ocr:8004/ocr/",
    "data_aggregation": "http://data-aggregation:8006/aggregate/",
}

@router.post("/process-cin/")
async def process_cin(file: UploadFile = File(...)):
    """
    Process CIN through all microservices and aggregate results.
    """
    try:
        # Read the file and encode it in Base64
        file_bytes = await file.read()
        file_base64 = base64.b64encode(file_bytes).decode("utf-8")

        # Step 1: Detect CIN
        async with httpx.AsyncClient(timeout=50.0) as client:
            detection_response = await client.post(
                MICROSERVICES["cin_detection"], json={"file": file_base64}
            )
        detection_response.raise_for_status()
        cropped_image_base64 = detection_response.json().get("cropped_image")
        if not cropped_image_base64:
            raise HTTPException(status_code=500, detail="Missing cropped_image from CIN detection")

        # Step 2: Extract fields
        async with httpx.AsyncClient() as client:
            extraction_response = await client.post(
                MICROSERVICES["field_extraction"], json={"file": cropped_image_base64}
            )
        extraction_response.raise_for_status()
        fields = extraction_response.json().get("fields")

        # Step 3: Preprocess fields
        preprocessed_fields = {}
        for field_name, field_base64 in fields.items():
            async with httpx.AsyncClient() as client:
                preprocess_response = await client.post(
                    MICROSERVICES["preprocessing"],
                    json={"file": field_base64, "field_name": field_name}
                )
            preprocess_response.raise_for_status()
            preprocessed_fields[field_name] = preprocess_response.json().get("preprocessed_image")

        # Step 4: Perform OCR
        extracted_text = {}
        for field_name, preprocessed_base64 in preprocessed_fields.items():
            async with httpx.AsyncClient() as client:
                ocr_response = await client.post(
                    MICROSERVICES["ocr"], json={"file": preprocessed_base64, "field_name": field_name}
                )
            ocr_response.raise_for_status()
            extracted_text[field_name] = ocr_response.json().get("text")

        # Step 5: Aggregate data
        async with httpx.AsyncClient() as client:
            aggregation_response = await client.post(
                MICROSERVICES["data_aggregation"], json=extracted_text
            )
        aggregation_response.raise_for_status()

        return aggregation_response.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Microservice request error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
