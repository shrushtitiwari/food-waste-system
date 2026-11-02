"""
Flask Web Application Entry Point & REST API Service.
Smart Food Waste Management System Prototype.
"""

from flask import Flask, render_template, jsonify, request
from services.smart_system import SmartFoodSystem

app = Flask(__name__)

# Initialize single global instance of Smart Food System service
smart_system = SmartFoodSystem()


@app.route("/")
def index():
    """Render main interactive dashboard page."""
    return render_template("index.html")


@app.route("/api/dashboard", methods=["GET"])
def get_dashboard_data():
    """Get complete dashboard state including DSA views, ML predictions, and KPI metrics."""
    data = smart_system.get_full_dashboard_state()
    return jsonify({"success": True, "data": data})


@app.route("/api/smart-analysis", methods=["POST"])
def trigger_smart_analysis():
    """Explicitly trigger the AI + DSA workflow execution."""
    summary = smart_system.run_smart_analysis()
    data = smart_system.get_full_dashboard_state()
    return jsonify({
        "success": True,
        "message": "Smart Analysis completed successfully! Inventory predictions, Max Heap priorities, and FIFO Queue updated.",
        "data": data
    })


@app.route("/api/items/add", methods=["POST"])
def add_item():
    """Add new item to Linked List inventory."""
    try:
        req_data = request.json
        name = req_data.get("name")
        category = req_data.get("category")
        quantity = float(req_data.get("quantity", 1.0))
        unit = req_data.get("unit", "kg")
        purchase_date = req_data.get("purchase_date")
        expiry_date = req_data.get("expiry_date")
        storage_type = req_data.get("storage_type", "Fridge")

        if not name or not category or not purchase_date or not expiry_date:
            return jsonify({"success": False, "error": "Missing required fields."}), 400

        added_item = smart_system.add_food_item(
            name=name,
            category=category,
            quantity=quantity,
            unit=unit,
            purchase_date=purchase_date,
            expiry_date=expiry_date,
            storage_type=storage_type
        )
        return jsonify({"success": True, "message": f"Added '{name}' to Linked List inventory!", "item": added_item})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/items/<item_id>", methods=["DELETE"])
def delete_item(item_id):
    """Delete item from Linked List inventory."""
    deleted = smart_system.remove_food_item(item_id)
    if deleted:
        return jsonify({"success": True, "message": "Item deleted successfully."})
    return jsonify({"success": False, "error": "Item not found."}), 404


@app.route("/api/items/<item_id>/consume", methods=["POST"])
def consume_item(item_id):
    """Consume item quantity."""
    req_data = request.json or {}
    consume_qty = req_data.get("quantity")
    if consume_qty is not None:
        consume_qty = float(consume_qty)

    consumed = smart_system.consume_item(item_id, consume_qty)
    if consumed:
        return jsonify({"success": True, "message": "Item consumption recorded!"})
    return jsonify({"success": False, "error": "Item not found."}), 404


@app.route("/api/queue/dequeue", methods=["POST"])
def dequeue_item():
    """Dequeue front item from FIFO Queue."""
    dequeued_node = smart_system.redistribution_queue.dequeue()
    if dequeued_node:
        smart_system.run_smart_analysis()
        return jsonify({
            "success": True,
            "message": f"Dequeued item '{dequeued_node.data.name}' for action: '{dequeued_node.action_type}'.",
            "item": dequeued_node.data.to_dict()
        })
    return jsonify({"success": False, "error": "Queue is currently empty."}), 400


@app.route("/api/reset-demo", methods=["POST"])
def reset_demo():
    """Reset inventory back to fresh synthetic demo batch."""
    smart_system.seed_demo_inventory()
    data = smart_system.get_full_dashboard_state()
    return jsonify({"success": True, "message": "Inventory reset to synthetic demo dataset.", "data": data})


if __name__ == "__main__":
    print("Starting Smart Food Waste Management System on http://0.0.0.0:5000 (accessible locally & via local Wi-Fi)")
    app.run(host="0.0.0.0", port=5000, debug=True)
