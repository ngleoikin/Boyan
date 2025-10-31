.PHONY: bootstrap gate brain ocr smoke stop

bootstrap:
	bash scripts/bootstrap_ubuntu.sh

gate:
	bash scripts/run_gate.sh

brain:
	bash scripts/run_brain.sh

ocr:
	bash scripts/run_ocr.sh

smoke:
	bash scripts/smoke.sh

stop:
	pkill -f "uvicorn .*mcp_gateway_face" || true; pkill -f "uvicorn .*brain_server" || true
