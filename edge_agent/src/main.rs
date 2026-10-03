use edge_agent::tools::{self, McpToolError};
use rmcp::{handler::server::wrapper::Parameters, tool_router, ServiceExt, transport::stdio};

#[derive(Clone)]
struct EdgeAgent;

#[tool_router(server_handler)]
impl EdgeAgent {
    #[tool(description = "Run inference on microgrid telemetry to get routing decision, throttle level, and trip hazard probability, then execute hardware action")]
    async fn evaluate(
        &self,
        Parameters(params): Parameters<edge_agent::tools::TelemetryParams>,
    ) -> Result<edge_agent::tools::ToolResult, rmcp::ErrorData> {
        let result = match tools::run_inference(params.telemetry).await {
            Ok(t) => t,
            Err(e) => match e {
                McpToolError::InvalidParams(msg) => {
                    return Err(rmcp::ErrorData::invalid_params(&msg, None));
                }
                McpToolError::GateTriggered(res) => {
                    // Still surface the data but with an error code; the client can
                    // render it if it wishes. Use a generic internal error so the
                    // request isn't rejected with 422 for a run-time policy.
                    return Err(rmcp::ErrorData::internal_error(
                        format!("safety gate triggered: {:?}", res.action),
                        None,
                    ));
                }
                McpToolError::HardwareFailure(msg) => {
                    return Err(rmcp::ErrorData::internal_error(
                        format!("hardware actuation failed: {}", msg),
                        None,
                    ));
                }
                McpToolError::Internal(msg) => {
                    return Err(rmcp::ErrorData::internal_error(&msg, None));
                }
            },
        };

        Ok(result)
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Serve over stdin/stdout (stdio transport)
    let service = EdgeAgent.serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}