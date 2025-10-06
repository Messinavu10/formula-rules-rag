"""
Performance tests for FIA Agent system.
"""

import time

import pytest


class TestPerformance:
    """Performance test cases for FIA Agent."""

    @pytest.mark.performance
    def test_agent_query_performance(self, fia_agent):
        """Test agent query response time."""
        question = "What are the safety requirements for Formula 1 cars?"

        start_time = time.time()
        response = fia_agent.query(question)
        end_time = time.time()

        response_time = end_time - start_time

        # Assert response time is reasonable (less than 30 seconds)
        assert response_time < 30.0, f"Query took too long: {response_time:.2f}s"

        # Assert response is valid
        assert "answer" in response
        assert response["answer"] is not None

        print(f"Query response time: {response_time:.2f}s")

    @pytest.mark.performance
    def test_multi_tool_performance(self, fia_agent):
        """Test multi-tool query performance."""
        question = "What are the safety requirements and penalties for violations?"

        start_time = time.time()
        response = fia_agent.query(question)
        end_time = time.time()

        response_time = end_time - start_time

        # Multi-tool queries should be faster than 45 seconds
        assert (
            response_time < 45.0
        ), f"Multi-tool query took too long: {response_time:.2f}s"

        # Assert response is valid
        assert "answer" in response
        assert response["answer"] is not None

        print(f"Multi-tool query response time: {response_time:.2f}s")

    @pytest.mark.performance
    def test_intent_classification_performance(self, fia_agent):
        """Test intent classification speed."""
        questions = [
            "What are the safety requirements?",
            "Compare Article 5 between 2024 and 2025",
            "What are the penalties for track limits?",
            "What is the weather today?",
        ]

        total_time = 0
        for question in questions:
            start_time = time.time()
            intent = fia_agent._classify_intent(question)
            end_time = time.time()

            classification_time = end_time - start_time
            total_time += classification_time

            # Each classification should be fast (less than 5 seconds)
            assert (
                classification_time < 5.0
            ), f"Intent classification too slow: {classification_time:.2f}s"

            # Assert intent is valid
            valid_intents = [
                "COMPARISON",
                "PENALTY",
                "SEARCH",
                "SUMMARY",
                "GENERAL",
                "MULTI_TOOL",
                "OUT_OF_SCOPE",
            ]
            assert intent in valid_intents, f"Invalid intent: {intent}"

        avg_time = total_time / len(questions)
        print(f"Average intent classification time: {avg_time:.2f}s")

        # Average should be reasonable
        assert avg_time < 3.0, f"Average classification time too slow: {avg_time:.2f}s"

    @pytest.mark.performance
    def test_tool_selection_performance(self, fia_agent):
        """Test tool selection speed."""
        intents = [
            "SEARCH",
            "COMPARISON",
            "PENALTY",
            "SUMMARY",
            "GENERAL",
            "OUT_OF_SCOPE",
        ]

        total_time = 0
        for intent in intents:
            start_time = time.time()
            tool = fia_agent._select_tool(intent)
            end_time = time.time()

            selection_time = end_time - start_time
            total_time += selection_time

            # Tool selection should be very fast (less than 1 second)
            assert (
                selection_time < 1.0
            ), f"Tool selection too slow: {selection_time:.2f}s"

            # Assert tool is valid
            valid_tools = [
                "regulation_search",
                "regulation_comparison",
                "penalty_lookup",
                "regulation_summary",
                "general_rag",
                "out_of_scope_handler",
            ]
            assert tool in valid_tools, f"Invalid tool: {tool}"

        avg_time = total_time / len(intents)
        print(f"Average tool selection time: {avg_time:.2f}s")

        # Average should be very fast
        assert avg_time < 0.5, f"Average tool selection time too slow: {avg_time:.2f}s"

    @pytest.mark.performance
    def test_memory_usage(self, fia_agent):
        """Test memory usage during agent operations."""
        import os

        import psutil

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run multiple queries
        questions = [
            "What are the safety requirements?",
            "What are the engine specifications?",
            "What are the penalties for violations?",
            "Compare regulations between years",
        ]

        for question in questions:
            response = fia_agent.query(question)
            assert "answer" in response

        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(f"Memory usage increase: {memory_increase:.2f} MB")

        # Memory increase should be reasonable (less than 100 MB)
        assert (
            memory_increase < 100.0
        ), f"Memory usage increased too much: {memory_increase:.2f} MB"

    @pytest.mark.performance
    def test_concurrent_queries(self, fia_agent):
        """Test system under concurrent load."""
        import queue
        import threading

        def query_worker(question, results_queue):
            """Worker function for concurrent queries."""
            try:
                start_time = time.time()
                response = fia_agent.query(question)
                end_time = time.time()

                results_queue.put(
                    {
                        "success": True,
                        "response_time": end_time - start_time,
                        "response": response,
                    }
                )
            except Exception as e:
                results_queue.put({"success": False, "error": str(e)})

        # Test concurrent queries
        questions = [
            "What are the safety requirements?",
            "What are the engine specifications?",
            "What are the penalties for violations?",
            "What is the weather today?",
        ]

        results_queue = queue.Queue()
        threads = []

        # Start concurrent queries
        start_time = time.time()
        for question in questions:
            thread = threading.Thread(
                target=query_worker, args=(question, results_queue)
            )
            thread.start()
            threads.append(thread)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        end_time = time.time()
        total_time = end_time - start_time

        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        # Assert all queries succeeded
        successful_queries = [r for r in results if r["success"]]
        assert len(successful_queries) == len(
            questions
        ), f"Only {len(successful_queries)}/{len(questions)} queries succeeded"

        # Assert total time is reasonable
        assert total_time < 60.0, f"Concurrent queries took too long: {total_time:.2f}s"

        # Calculate average response time
        response_times = [r["response_time"] for r in successful_queries]
        avg_response_time = sum(response_times) / len(response_times)

        print(f"Concurrent queries total time: {total_time:.2f}s")
        print(f"Average response time: {avg_response_time:.2f}s")

        # Average response time should be reasonable
        assert (
            avg_response_time < 20.0
        ), f"Average response time too slow: {avg_response_time:.2f}s"
