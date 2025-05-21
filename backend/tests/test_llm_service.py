import unittest
from unittest.mock import patch, MagicMock, AsyncMock # AsyncMock might be needed for generate_content
import asyncio
import decimal # For comparing confidence values if they are Decimal

# Adjust the import path based on the project structure and how tests are run
# Assuming 'backend/src' is in PYTHONPATH or tests are run from a level where this path is valid.
# If running tests from project root with 'python -m unittest discover backend',
# then backend.src... might not be found directly without PYTHONPATH manipulation.
# For now, let's assume a common structure where 'src' is a root for the package.
# This was handled in temporary_manual_test.py by sys.path.append.
# For unit tests, this is often handled by test runner configuration or project structure.
# Let's try a relative path that might work if tests are run from 'backend' dir or if 'backend' is a package.
# This might need adjustment.
from backend_api.llm_service.service import LLMService

# Helper to create mock LLM stream parts (text content for the part)
def create_mock_llm_response_part(text_content: str):
    part = MagicMock()
    part.text = text_content # This should be a string, LLMService will encode it

    candidate = MagicMock()
    candidate.content.parts = [part]
    
    # Simulate the structure of GenerationResponse
    response_chunk = MagicMock() 
    response_chunk.candidates = [candidate]
    return response_chunk

# Async generator to feed to the mocked generate_content
async def mock_response_stream_generator(*text_contents: str):
    for text_content in text_contents:
        yield create_mock_llm_response_part(text_content)
        await asyncio.sleep(0) # Yield control, simulate async behavior

class TestLLMServiceStreamAnswer(unittest.IsolatedAsyncioTestCase):

    # Patch vertexai.init and GenerativeModel for all tests in this class
    # These patches are applied to where the names are looked up (the service module)
    @patch('backend_api.llm_service.service.vertexai.init')
    @patch('backend_api.llm_service.service.GenerativeModel')
    async def test_normal_streaming_high_confidence(self, MockGenerativeModel, mock_vertex_init):
        # Configure the mock for GenerativeModel instance
        mock_model_instance = MockGenerativeModel.return_value
        
        # Configure generate_content to be a MagicMock returning our async generator
        # because generate_content(stream=True) is a sync method returning an AsyncIterable
        mock_model_instance.generate_content = MagicMock(
            return_value=mock_response_stream_generator(
                '{"confidence": 0.95, "answer": "Paris is the capital of France. ',
                'It is known for the Eiffel Tower.", "source_pages": [1, 2]}'
            )
        )

        llm_service = LLMService(model_name="test-model")
        
        query = "What is the capital of France?"
        context_chunks = [{"metadata": {"page": 1}, "content": "Some context."}]
        
        results = []
        async for result in llm_service.stream_answer(query, context_chunks):
            results.append(result)

        # Assertions
        self.assertTrue(len(results) > 0, "Should have received some results.")
        
        # Check for chunk messages
        # Based on the ijson behavior, the full answer string is often yielded in one chunk.
        chunk_found = False
        for r in results:
            if r['type'] == 'chunk':
                self.assertIn("Paris is the capital of France.", r['content'])
                self.assertIn("Eiffel Tower", r['content'])
                chunk_found = True
                break
        self.assertTrue(chunk_found, "Answer chunk not found or content mismatch.")

        # Check for final message
        final_msg = next((r for r in results if r['type'] == 'final'), None)
        self.assertIsNotNone(final_msg, "Final message not found.")
        
        if final_msg: # Keep linters happy
            self.assertEqual(final_msg['content']['answer'], "Paris is the capital of France. It is known for the Eiffel Tower.")
            # LLMService uses Decimal for confidence if ijson parses it as such.
            # Let's ensure comparison is type-agnostic or cast to float.
            self.assertAlmostEqual(float(final_msg['content']['confidence']), 0.95, places=2)
            self.assertEqual(final_msg['content']['source_pages'], [1, 2])

    @patch('backend_api.llm_service.service.vertexai.init')
    @patch('backend_api.llm_service.service.GenerativeModel')
    async def test_low_confidence(self, MockGenerativeModel, mock_vertex_init):
        mock_model_instance = MockGenerativeModel.return_value
        mock_model_instance.generate_content = MagicMock(
            return_value=mock_response_stream_generator(
                '{"confidence": 0.1, "answer": "Not sure.", "source_pages": [3]}'
            )
        )

        llm_service = LLMService(model_name="test-model")
        results = []
        async for result in llm_service.stream_answer("test query", []):
            results.append(result)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['type'], 'error')
        self.assertEqual(results[0]['content'], 'Confidence too low')

    @patch('backend_api.llm_service.service.vertexai.init')
    @patch('backend_api.llm_service.service.GenerativeModel')
    async def test_malformed_json(self, MockGenerativeModel, mock_vertex_init):
        mock_model_instance = MockGenerativeModel.return_value
        mock_model_instance.generate_content = MagicMock(
            return_value=mock_response_stream_generator(
                '{"confidence": 0.9, "answer": "Incomplete... ' # Missing closing brace and quote
            )
        )

        llm_service = LLMService(model_name="test-model")
        results = []
        async for result in llm_service.stream_answer("test query", []):
            results.append(result)
            
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['type'], 'error')
        # The exact message comes from the IncompleteJSONError
        self.assertIn('Could not get source pages due to incomplete JSON', results[0]['content'])


    @patch('backend_api.llm_service.service.vertexai.init')
    @patch('backend_api.llm_service.service.GenerativeModel')
    async def test_missing_confidence(self, MockGenerativeModel, mock_vertex_init):
        mock_model_instance = MockGenerativeModel.return_value
        mock_model_instance.generate_content = MagicMock(
            return_value=mock_response_stream_generator(
                '{"answer": "Answer without confidence.", "source_pages": [1]}'
            )
        )

        llm_service = LLMService(model_name="test-model")
        results = []
        async for result in llm_service.stream_answer("test query", []):
            results.append(result)
            
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['type'], 'error')
        self.assertEqual(results[0]['content'], 'Could not determine confidence')


    @patch('backend_api.llm_service.service.vertexai.init')
    @patch('backend_api.llm_service.service.GenerativeModel')
    async def test_missing_answer(self, MockGenerativeModel, mock_vertex_init):
        mock_model_instance = MockGenerativeModel.return_value
        mock_model_instance.generate_content = MagicMock(
            return_value=mock_response_stream_generator(
                '{"confidence": 0.8, "source_pages": [1]}' # Missing "answer"
            )
        )

        llm_service = LLMService(model_name="test-model")
        results = []
        async for result in llm_service.stream_answer("test query", []):
            results.append(result)
            
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['type'], 'error')
        self.assertEqual(results[0]['content'], 'Could not get source pages')


    @patch('backend_api.llm_service.service.GenerativeModel') # Only one patch needed if vertexai.init is not called
    @patch('backend_api.llm_service.service.vertexai.init') # Keep vertexai.init patch for consistency
    async def test_missing_source_pages(self, mock_vertex_init, MockGenerativeModel): # Order of args matters for patches
        mock_model_instance = MockGenerativeModel.return_value
        mock_model_instance.generate_content = MagicMock(
            return_value=mock_response_stream_generator(
                '{"confidence": 0.8, "answer": "Answer without source pages."}' # Missing "source_pages"
            )
        )

        llm_service = LLMService(model_name="test-model")
        results = []
        async for result in llm_service.stream_answer("test query", []):
            results.append(result)
            
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['type'], 'error')
        self.assertEqual(results[0]['content'], 'Could not get source pages')

if __name__ == '__main__':
    # This allows running the tests directly with `python backend/tests/test_llm_service.py`
    # But usually, a test runner like `pytest` or `python -m unittest discover` is used.
    # For `unittest.IsolatedAsyncioTestCase` to work correctly when run directly,
    # it's often better to use `unittest.main()`.
    unittest.main()
