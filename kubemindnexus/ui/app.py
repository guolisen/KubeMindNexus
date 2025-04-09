"""Streamlit UI for KubeMindNexus."""
import asyncio
import json
import time
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Callable

import httpx
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from ..config.config import get_config
from ..utils.logger import LoggerMixin


class StreamlitUI(LoggerMixin):
    """Streamlit UI for KubeMindNexus."""
    
    def __init__(self, api_base_url: Optional[str] = None) -> None:
        """Initialize the Streamlit UI.
        
        Args:
            api_base_url: The base URL for the API. If None, uses the one from config.
        """
        config = get_config()
        self.api_base_url = api_base_url or config.get_api_base_url()
        
        # Initialize session state
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        
        if "clusters" not in st.session_state:
            st.session_state.clusters = []
        
        if "servers" not in st.session_state:
            st.session_state.servers = []
        
        if "current_cluster" not in st.session_state:
            st.session_state.current_cluster = None
        
        if "llm_providers" not in st.session_state:
            st.session_state.llm_providers = []
        
        if "current_provider" not in st.session_state:
            st.session_state.current_provider = None
    
    def run(self) -> None:
        """Run the Streamlit UI."""
        config = get_config()
        ui_title = config.get_ui_title()

        st.set_page_config(
            page_title=ui_title,
            page_icon="🔮",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title(ui_title)
        
        # Sidebar for configuration
        with st.sidebar:
            st.title("Configuration")
            
            # Refresh button
            if st.button("Refresh Data"):
                self._load_data()
            
            # LLM provider selection
            st.subheader("LLM Provider")
            self._render_llm_selector()
            
            # Cluster management
            st.subheader("Clusters")
            self._render_cluster_manager()
            
            # MCP server management
            st.subheader("MCP Servers")
            self._render_server_manager()
            
            # System status
            st.subheader("System Status")
            self._render_system_status()
        
        # Main content area with tabs
        tab1, tab2, tab3 = st.tabs(["Chat", "Cluster Status", "Chat History"])
        
        # Chat tab
        with tab1:
            self._render_chat_interface()
        
        # Cluster status tab
        with tab2:
            self._render_cluster_status()
        
        # Chat history tab
        with tab3:
            self._render_chat_history()
    
    def _load_data(self) -> None:
        """Load initial data from the API."""
        try:
            # Load clusters
            clusters = self._api_get("/api/clusters")
            st.session_state.clusters = clusters
            
            # Load servers
            servers = self._api_get("/api/mcp/servers")
            st.session_state.servers = servers
            
            # Load LLM providers
            providers = self._api_get("/api/llm/providers")
            st.session_state.llm_providers = providers
            
            # Set default provider
            for provider in providers:
                if provider.get("is_default", False):
                    st.session_state.current_provider = provider["name"]
                    break
        
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            st.error(f"Error loading data: {str(e)}")
    
    def _render_llm_selector(self) -> None:
        """Render the LLM provider selector."""
        if not st.session_state.llm_providers:
            try:
                providers = self._api_get("/api/llm/providers")
                st.session_state.llm_providers = providers
                
                # Set default provider
                for provider in providers:
                    if provider.get("is_default", False):
                        st.session_state.current_provider = provider["name"]
                        break
            except Exception as e:
                st.error(f"Error loading LLM providers: {str(e)}")
                return
        
        provider_names = [p["name"] for p in st.session_state.llm_providers]
        current_provider = st.session_state.current_provider or (
            provider_names[0] if provider_names else None
        )
        
        selected_provider = st.selectbox(
            "Select LLM Provider",
            options=provider_names,
            index=provider_names.index(current_provider) if current_provider in provider_names else 0,
            key="llm_provider_selectbox"
        )
        
        if selected_provider != st.session_state.current_provider:
            st.session_state.current_provider = selected_provider
            try:
                self._api_post(f"/api/llm/providers/{selected_provider}/default")
                st.success(f"Set {selected_provider} as default provider")
            except Exception as e:
                st.error(f"Error setting default provider: {str(e)}")
    
    def _render_cluster_manager(self) -> None:
        """Render the cluster manager interface."""
        # Load clusters if not loaded
        if not st.session_state.clusters:
            try:
                clusters = self._api_get("/api/clusters")
                st.session_state.clusters = clusters
            except Exception as e:
                st.error(f"Error loading clusters: {str(e)}")
                return
        
        # Cluster selector
        cluster_names = ["None"] + [c["name"] for c in st.session_state.clusters]
        current_index = 0
        if st.session_state.current_cluster:
            try:
                current_index = cluster_names.index(st.session_state.current_cluster)
            except ValueError:
                current_index = 0
        
        selected_cluster = st.selectbox(
            "Select Cluster",
            options=cluster_names,
            index=current_index,
            key="cluster_selectbox"
        )
        
        if selected_cluster == "None":
            st.session_state.current_cluster = None
        else:
            st.session_state.current_cluster = selected_cluster
        
        # Connect/disconnect button
        if st.session_state.current_cluster:
            # Find cluster in state
            cluster = next(
                (c for c in st.session_state.clusters if c["name"] == st.session_state.current_cluster),
                None
            )
            
            if cluster:
                status = cluster.get("status", "unknown")
                
                if status == "connected":
                    if st.button("Disconnect"):
                        try:
                            self._api_post(f"/api/clusters/{cluster['name']}/disconnect")
                            st.success(f"Disconnected from cluster {cluster['name']}")
                            # Refresh cluster data
                            clusters = self._api_get("/api/clusters")
                            st.session_state.clusters = clusters
                        except Exception as e:
                            st.error(f"Error disconnecting from cluster: {str(e)}")
                else:
                    if st.button("Connect"):
                        try:
                            self._api_post(f"/api/clusters/{cluster['name']}/connect")
                            st.success(f"Connected to cluster {cluster['name']}")
                            # Refresh cluster data
                            clusters = self._api_get("/api/clusters")
                            st.session_state.clusters = clusters
                        except Exception as e:
                            st.error(f"Error connecting to cluster: {str(e)}")
        
        # Cluster creation form
        with st.expander("Add New Cluster"):
            with st.form("add_cluster_form"):
                cluster_name = st.text_input("Cluster Name")
                cluster_ip = st.text_input("IP Address")
                cluster_port = st.number_input("Port", min_value=1, max_value=65535, value=6443)
                cluster_description = st.text_area("Description")
                
                submitted = st.form_submit_button("Add Cluster")
                if submitted:
                    if not cluster_name or not cluster_ip:
                        st.error("Cluster name and IP address are required")
                    else:
                        try:
                            self._api_post(
                                "/api/clusters",
                                {
                                    "name": cluster_name,
                                    "ip": cluster_ip,
                                    "port": cluster_port,
                                    "description": cluster_description
                                }
                            )
                            st.success(f"Added cluster {cluster_name}")
                            # Refresh cluster data
                            clusters = self._api_get("/api/clusters")
                            st.session_state.clusters = clusters
                        except Exception as e:
                            st.error(f"Error adding cluster: {str(e)}")
        
        # Cluster deletion
        if st.session_state.current_cluster:
            with st.expander("Delete Current Cluster"):
                st.warning(f"Are you sure you want to delete cluster {st.session_state.current_cluster}?")
                if st.button("Delete Cluster"):
                    try:
                        self._api_delete(f"/api/clusters/{st.session_state.current_cluster}")
                        st.success(f"Deleted cluster {st.session_state.current_cluster}")
                        st.session_state.current_cluster = None
                        # Refresh cluster data
                        clusters = self._api_get("/api/clusters")
                        st.session_state.clusters = clusters
                    except Exception as e:
                        st.error(f"Error deleting cluster: {str(e)}")
    
    def _render_server_manager(self) -> None:
        """Render the MCP server manager interface."""
        # Server creation form
        with st.expander("Add New MCP Server"):
            with st.form("add_server_form"):
                server_name = st.text_input("Server Name")
                server_type = st.selectbox(
                    "Server Type",
                    options=["stdio", "sse"]
                )
                
                is_local = st.checkbox("Local Server")
                
                if not is_local:
                    # For non-local servers, require a cluster
                    cluster_names = [c["name"] for c in st.session_state.clusters]
                    server_cluster = st.selectbox(
                        "Cluster",
                        options=cluster_names if cluster_names else ["No clusters available"]
                    )
                
                if server_type == "stdio":
                    server_command = st.text_input("Command")
                    server_args = st.text_area("Arguments (one per line)")
                    server_env = st.text_area("Environment Variables (KEY=VALUE, one per line)")
                    server_url = None
                else:  # sse
                    server_command = None
                    server_args = None
                    server_env = None
                    server_url = st.text_input("URL")
                
                submitted = st.form_submit_button("Add Server")
                if submitted:
                    if not server_name:
                        st.error("Server name is required")
                    elif server_type == "stdio" and not server_command:
                        st.error("Command is required for stdio servers")
                    elif server_type == "sse" and not server_url:
                        st.error("URL is required for SSE servers")
                    elif not is_local and (not cluster_names or server_cluster == "No clusters available"):
                        st.error("No clusters available for non-local servers")
                    else:
                        # Parse args and env
                        args_list = []
                        if server_args:
                            args_list = [arg.strip() for arg in server_args.split("\n") if arg.strip()]
                        
                        env_dict = {}
                        if server_env:
                            for line in server_env.split("\n"):
                                line = line.strip()
                                if line and "=" in line:
                                    key, value = line.split("=", 1)
                                    env_dict[key.strip()] = value.strip()
                        
                        # Create server
                        try:
                            endpoint = "/api/mcp/servers"
                            if not is_local:
                                endpoint += f"?cluster_name={server_cluster}"
                            
                            self._api_post(
                                endpoint,
                                {
                                    "name": server_name,
                                    "type": server_type,
                                    "command": server_command,
                                    "args": args_list,
                                    "env": env_dict,
                                    "url": server_url,
                                    "is_local": is_local
                                }
                            )
                            st.success(f"Added server {server_name}")
                            # Refresh server data
                            servers = self._api_get("/api/mcp/servers")
                            st.session_state.servers = servers
                        except Exception as e:
                            st.error(f"Error adding server: {str(e)}")
    
    def _render_system_status(self) -> None:
        """Render the system status."""
        try:
            health = self._api_get("/api/health")
            
            # Overall status
            st.write(f"Status: {health['status']}")
            
            # API status
            st.write(f"API: {health['api']['status']}")
            
            # Database status
            st.write(f"Database: {health['api']['database']}")
            
            # Clusters status
            if health["clusters"]:
                st.write("Clusters:")
                for name, status in health["clusters"].items():
                    st.write(f"- {name}: {status['status']}")
            else:
                st.write("No clusters connected")
        except Exception as e:
            st.error(f"Error getting system status: {str(e)}")
    
    def _render_chat_interface(self) -> None:
        """Render the chat interface."""
        st.subheader("Chat with Clusters")
        
        # Display chat messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about your Kubernetes clusters..."):
            # Add user message to chat
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get response from API
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Prepare request
                        request_data = {
                            "message": prompt,
                            "llm_provider": st.session_state.current_provider
                        }
                        
                        # Add cluster name if selected
                        if st.session_state.current_cluster:
                            request_data["cluster_name"] = st.session_state.current_cluster
                        
                        # Make API call
                        response = self._api_post("/api/chat", request_data)
                        
                        # Display response
                        st.write(response["message"])
                        
                        # Add to chat history
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": response["message"]
                        })
                    
                    except Exception as e:
                        error_message = f"Error: {str(e)}"
                        st.error(error_message)
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": error_message
                        })
    
    def _render_cluster_status(self) -> None:
        """Render the cluster status dashboard."""
        st.subheader("Cluster Status Dashboard")
        
        # Display status for each cluster
        if not st.session_state.clusters:
            st.info("No clusters available. Add a cluster in the sidebar.")
            return
        
        # Create a grid for displaying clusters
        cols = st.columns(min(3, len(st.session_state.clusters)))
        
        for i, cluster in enumerate(st.session_state.clusters):
            col = cols[i % len(cols)]
            
            with col:
                status = cluster.get("status", "unknown")
                status_color = {
                    "connected": "green",
                    "created": "blue",
                    "not_connected": "orange",
                    "unknown": "gray"
                }.get(status, "gray")
                
                st.markdown(f"""
                <div style="border:1px solid {status_color}; border-radius:5px; padding:10px; margin-bottom:10px;">
                    <h3 style="color:{status_color};">{cluster["name"]}</h3>
                    <p>IP: {cluster["ip"]}:{cluster["port"]}</p>
                    <p>Status: <span style="color:{status_color};">{status}</span></p>
                    <p>{cluster.get("description", "")}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Only show detailed status for connected clusters
                if status == "connected":
                    if st.button(f"View Details ({cluster['name']})", key=f"view_details_{cluster['name']}"):
                        try:
                            cluster_status = self._api_get(f"/api/clusters/{cluster['name']}/status")
                            
                            # Display detailed status
                            st.json(cluster_status)
                        except Exception as e:
                            st.error(f"Error getting cluster status: {str(e)}")
    
    def _render_chat_history(self) -> None:
        """Render the chat history."""
        st.subheader("Chat History")
        
        try:
            history = self._api_get("/api/chat/history")
            
            if not history:
                st.info("No chat history available.")
                return
            
            # Display each chat entry
            for entry in history:
                with st.expander(
                    f"{entry['timestamp']} - LLM: {entry['llm_provider']}"
                    + (f" - Cluster: {entry['cluster_name']}" if entry.get('cluster_name') else "")
                ):
                    st.markdown("**User:**")
                    st.write(entry["user_message"])
                    st.markdown("**Assistant:**")
                    st.write(entry["assistant_response"])
        
        except Exception as e:
            st.error(f"Error getting chat history: {str(e)}")
    
    def _api_get(self, endpoint: str) -> Any:
        """Make a GET request to the API.
        
        Args:
            endpoint: The API endpoint to call.
            
        Returns:
            The response data.
            
        Raises:
            Exception: If the request fails.
        """
        url = f"{self.api_base_url}{endpoint}"
        
        try:
            response = httpx.get(url)
            response.raise_for_status()
            return response.json()
        
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise Exception(f"API error {e.response.status_code}: {e.response.text}")
        
        except Exception as e:
            self.logger.error(f"Request error: {str(e)}")
            raise Exception(f"Request error: {str(e)}")
    
    def _api_post(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Any:
        """Make a POST request to the API.
        
        Args:
            endpoint: The API endpoint to call.
            data: The data to send.
            
        Returns:
            The response data.
            
        Raises:
            Exception: If the request fails.
        """
        url = f"{self.api_base_url}{endpoint}"
        
        try:
            response = httpx.post(url, json=data)
            response.raise_for_status()
            return response.json()
        
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise Exception(f"API error {e.response.status_code}: {e.response.text}")
        
        except Exception as e:
            self.logger.error(f"Request error: {str(e)}")
            raise Exception(f"Request error: {str(e)}")
    
    def _api_delete(self, endpoint: str) -> None:
        """Make a DELETE request to the API.
        
        Args:
            endpoint: The API endpoint to call.
            
        Raises:
            Exception: If the request fails.
        """
        url = f"{self.api_base_url}{endpoint}"
        
        try:
            response = httpx.delete(url)
            response.raise_for_status()
        
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise Exception(f"API error {e.response.status_code}: {e.response.text}")
        
        except Exception as e:
            self.logger.error(f"Request error: {str(e)}")
            raise Exception(f"Request error: {str(e)}")


# Singleton instance
ui = StreamlitUI()


def main() -> None:
    """Run the Streamlit UI."""
    ui.run()


if __name__ == "__main__":
    main()
