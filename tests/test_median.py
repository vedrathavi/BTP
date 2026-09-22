import unittest

import torch

from algorithms.median import aggregate


class MedianAggregationTests(unittest.TestCase):
    def test_one_client(self):
        state = {"weight": torch.tensor([1.0, 2.0])}
        result, _ = aggregate([state], [10])
        torch.testing.assert_close(result["weight"], state["weight"])

    def test_two_clients_averages_middle_values(self):
        states = [
            {"weight": torch.tensor([1.0, 10.0])},
            {"weight": torch.tensor([3.0, 20.0])},
        ]
        result, _ = aggregate(states, [1, 1])
        torch.testing.assert_close(result["weight"], torch.tensor([2.0, 15.0]))

    def test_three_clients_selects_middle_value(self):
        states = [
            {"weight": torch.tensor([1.0, 10.0])},
            {"weight": torch.tensor([2.0, 20.0])},
            {"weight": torch.tensor([3.0, 30.0])},
        ]
        result, _ = aggregate(states, [1, 1, 1])
        torch.testing.assert_close(result["weight"], torch.tensor([2.0, 20.0]))

    def test_four_clients_averages_middle_values(self):
        states = [
            {"weight": torch.tensor([0.1])},
            {"weight": torch.tensor([0.2])},
            {"weight": torch.tensor([0.3])},
            {"weight": torch.tensor([100.0])},
        ]
        result, _ = aggregate(states, [1, 1, 1, 1])
        torch.testing.assert_close(result["weight"], torch.tensor([0.25]))

    def test_four_client_outlier(self):
        states = [
            {"weight": torch.tensor([1.0, 10.0])},
            {"weight": torch.tensor([2.0, 20.0])},
            {"weight": torch.tensor([3.0, 30.0])},
            {"weight": torch.tensor([1000.0, 40.0])},
        ]
        result, _ = aggregate(states, [1, 1, 1, 1])
        torch.testing.assert_close(result["weight"], torch.tensor([2.5, 25.0]))

    def test_multidimensional_tensors(self):
        states = [
            {"weight": torch.tensor([[1.0, 10.0], [100.0, 1000.0]])},
            {"weight": torch.tensor([[3.0, 30.0], [300.0, 3000.0]])},
        ]
        result, _ = aggregate(states, [1, 1])
        expected = torch.tensor([[2.0, 20.0], [200.0, 2000.0]])
        torch.testing.assert_close(result["weight"], expected)

    def test_non_floating_buffer_copies_first_client(self):
        states = [
            {
                "weight": torch.tensor([1.0]),
                "counter": torch.tensor(7, dtype=torch.int64),
            },
            {
                "weight": torch.tensor([3.0]),
                "counter": torch.tensor(9, dtype=torch.int64),
            },
        ]
        result, _ = aggregate(states, [1, 1])
        self.assertEqual(result["counter"].item(), 7)
        self.assertEqual(result["counter"].dtype, torch.int64)

    def test_empty_input(self):
        with self.assertRaisesRegex(ValueError, "local_weights cannot be empty"):
            aggregate([], [])


if __name__ == "__main__":
    unittest.main()
