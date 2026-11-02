"""
Automated Verification Test Suite for Smart Food Waste Management System.
Tests custom DSA implementations, ML predictor, and integrated system pipeline.
"""

import unittest
from datetime import date, timedelta
from ds.linked_list import FoodLinkedList, Node
from ds.max_heap import MaxHeap
from ds.fifo_queue import FIFOQueue
from ml.generator import SyntheticFoodDataGenerator
from ml.predictor import FoodPredictorModel
from services.smart_system import SmartFoodSystem


class TestFoodWasteSystem(unittest.TestCase):

    def test_custom_linked_list(self):
        """Test Linked List insertion, search, update, deletion, and chain format."""
        ll = FoodLinkedList()
        today = str(date.today())
        exp = str(date.today() + timedelta(days=5))

        # Add items
        node1 = ll.add_item("1", "Milk", "Dairy", 2.0, "L", today, exp)
        node2 = ll.add_item("2", "Rice", "Grains & Staples", 5.0, "kg", today, str(date.today() + timedelta(days=100)))
        node3 = ll.add_item("3", "Bread", "Bakery", 1.0, "units", today, str(date.today() + timedelta(days=2)))

        self.assertEqual(ll.size, 3)
        self.assertEqual(ll.visualize_chain(), "Milk -> Rice -> Bread -> NULL")

        # Search
        found = ll.search_items(query="Milk")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].name, "Milk")

        # Update
        updated = ll.update_item("1", quantity=1.5)
        self.assertTrue(updated)
        self.assertEqual(ll.find_by_id("1").quantity, 1.5)

        # Delete
        deleted = ll.delete_item("2")
        self.assertTrue(deleted)
        self.assertEqual(ll.size, 2)
        self.assertEqual(ll.visualize_chain(), "Milk -> Bread -> NULL")

    def test_custom_max_heap(self):
        """Test Max Heap priority ranking and heapify."""
        today = str(date.today())

        node_urgent = Node("1", "Bread", "Bakery", 1.0, "units", today, str(date.today() + timedelta(days=1)))
        node_urgent.waste_score = 90.0
        node_urgent.predicted_waste = 0.8
        node_urgent.days_to_expiry = 1

        node_stable = Node("2", "Rice", "Grains & Staples", 5.0, "kg", today, str(date.today() + timedelta(days=120)))
        node_stable.waste_score = 5.0
        node_stable.predicted_waste = 0.0
        node_stable.days_to_expiry = 120

        heap = MaxHeap()
        heap.heapify([node_stable, node_urgent])

        # Urgent node must be extracted first (root of Max Heap)
        extracted_root = heap.extract_max()
        self.assertIsNotNone(extracted_root)
        self.assertEqual(extracted_root.node.name, "Bread")

    def test_custom_fifo_queue(self):
        """Test FIFO Queue enqueue, dequeue, peek, size."""
        queue = FIFOQueue()
        node1 = Node("1", "Milk", "Dairy", 2.0, "L", "2026-09-20", "2026-09-25")
        node2 = Node("2", "Bread", "Bakery", 1.0, "units", "2026-09-20", "2026-09-23")

        queue.enqueue(node1, "Consume First")
        queue.enqueue(node2, "Redistribute")

        self.assertEqual(queue.size(), 2)
        self.assertEqual(queue.peek().name, "Milk")
        self.assertIn("FRONT -> Milk", queue.visualize_queue())

        dequeued = queue.dequeue()
        self.assertEqual(dequeued.data.name, "Milk")
        self.assertEqual(queue.size(), 1)
        self.assertEqual(queue.peek().name, "Bread")

    def test_ml_predictor(self):
        """Test ML training and node prediction inference."""
        predictor = FoodPredictorModel()
        self.assertTrue(predictor.is_trained)

        today = str(date.today())
        exp = str(date.today() + timedelta(days=2))
        node = Node("999", "Whole Milk", "Dairy", 3.0, "L", today, exp)

        prediction = predictor.predict_node(node)
        self.assertIn("waste_risk", prediction)
        self.assertIn("consumption_demand", prediction)
        self.assertIn("predicted_waste", prediction)
        self.assertGreaterEqual(node.waste_score, 0.0)

    def test_integrated_system_workflow(self):
        """Test end-to-end Smart Analysis workflow."""
        system = SmartFoodSystem()
        summary = system.run_smart_analysis()

        self.assertGreater(summary["total_items"], 0)
        self.assertIn("linked_list_visual", summary)
        self.assertIn("queue_visual", summary)

        dashboard_state = system.get_full_dashboard_state()
        self.assertIn("summary", dashboard_state)
        self.assertIn("inventory", dashboard_state)
        self.assertIn("priority_heap", dashboard_state)
        self.assertIn("redistribution_queue", dashboard_state)
        self.assertIn("recommendations", dashboard_state)


if __name__ == "__main__":
    unittest.main()
