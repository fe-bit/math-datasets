from math_datasets.datasets.base_dataset import Dataset
from math_datasets.generators import get_output_file, OllamaGenerate
import json
from tqdm import tqdm


class ResponseTransformator:
    """
    This class is responsible for transforming the response from the API into a more usable format.
    """
    def __init__(self):
        self.generator = OllamaGenerate("mistral:7B")

    def transform(self, dataset: Dataset, model_name: str, save_dir: str, overwrite: bool = False):
        """
        Transforms the response into a more usable format.
        """
        # Implement the transformation logic here
        output_path = get_output_file(save_dir, model_name, dataset)
        try:
            with open(output_path, "r") as f:
                entries = [json.loads(line) for line in f]            
        except FileNotFoundError:
            print(f"❌ {model_name}: No output file found for {dataset.name}.")
            return None
        
        print(f"🚀 {model_name}: Begin with Answer Transformation for {dataset.name}.")
        counter = 0
        for entry in tqdm(entries):
            # Extract the answer from the response
            if not overwrite and "extracted_response" in entry:
                # If the answer is already extracted, skip this entry
                continue

            answer = self._extract_answer(entry["response"])
            entry["extracted_response"] = answer
            counter += 1
            if counter == 5:
                # Rewrite!
                counter = 0
                with open(output_path, "w") as f:
                    for entry in entries:
                        # Transform the response
                        f.write(json.dumps(entry) + "\n")
        
        with open(output_path, "w") as f:
            for entry in entries:
                # Transform the response
                f.write(json.dumps(entry) + "\n")
        print(f"✅ {model_name}: Transformed entries in {dataset.name} dataset.")

    def _extract_answer(self, response: str) -> str:
        """
        Extracts the answer from the response.
        """
        # Implement the extraction logic here
        return self.generator.generate(
            "Extract the answer from the following response: " + response, {}
        )