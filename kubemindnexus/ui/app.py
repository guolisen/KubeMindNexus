"""Streamlit UI for KubeMindNexus."""

import asyncio
import json
import logging
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

from ..config import Configuration
from ..constants import LLMProvider
from ..database import DatabaseManager
from ..llm.base import BaseLLM, LLMFactory
from ..llm.react import ReactLoop
from ..mcp.hub import MCPHub

logger = logging.getLogger(__name__)


class StreamlitApp:
    """Streamlit UI application for KubeMindNexus."""
    
    def __init__(
        self,
        config: Configuration,
        db_manager: DatabaseManager,
        mcp_hub: MCPHub,
        react_loop: ReactLoop,
    ):
        """Initialize Streamlit application.
        
        Args:
            config: Configuration instance.
            db_manager: Database manager instance.
            mcp_hub: MCP hub instance.
            react_loop: ReactLoop instance.
        """
        self.config = config
        self.db_manager = db_manager
        self.mcp_hub = mcp_hub
        self.react_loop = react_loop
        
        # Custom CSS for UI
        self._apply_custom_styles()
        
        # Initialize session state
        self._init_session_state()
    
    def _apply_custom_styles(self):
        """Apply custom CSS styles to the app."""
        st.markdown(
            """
            <style>
            .main {
                max-width: 1200px;
                padding: 2rem;
            }
            .chat-message {
                padding: 1rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                display: flex;
                flex-direction: column;
            }
            .chat-message.user {
                background-color: #f0f2f6;
            }
            .chat-message.bot {
                background-color: #def3ff;
            }
            .chat-bubble {
                padding: 0.5rem 1rem;
                border-radius: 0.5rem;
                margin-bottom: 0.5rem;
                max-width: 80%;
            }
            .user-bubble {
                background-color: #2970FF;
                color: white;
                align-self: flex-end;
            }
            .bot-bubble {
                background-color: #F0F0F0;
                color: black;
                align-self: flex-start;
            }
            .cluster-metrics {
                padding: 1rem;
                border: 1px solid #ddd;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
            }
            .metrics-header {
                font-size: 1.2rem;
                font-weight: bold;
                margin-bottom: 0.5rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    
    def _init_session_state(self):
        """Initialize Streamlit session state variables."""
        if "current_page" not in st.session_state:
            st.session_state.current_page = "Chat"
            
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            
        if "thinking" not in st.session_state:
            st.session_state.thinking = False
            
        if "current_cluster" not in st.session_state:
            st.session_state.current_cluster = None
    
    def run(self):
        """Run the Streamlit application."""
        st.title("KubeMindNexus")
        st.caption("Kubernetes Clusters Management with Model Context Protocol")
        
        # App navigation sidebar
        self._render_sidebar()
        
        # Main content based on current page
        if st.session_state.current_page == "Chat":
            self._render_chat_page()
        elif st.session_state.current_page == "Clusters":
            self._render_clusters_page()
        elif st.session_state.current_page == "MCP Servers":
            self._render_mcp_servers_page()
    
    def _render_sidebar(self):
        """Render the application sidebar."""
        with st.sidebar:
            st.title("Navigation")
            
            # Navigation options
            if st.button("Chat", use_container_width=True):
                st.session_state.current_page = "Chat"
                st.rerun()
                
            if st.button("Clusters", use_container_width=True):
                st.session_state.current_page = "Clusters"
                st.rerun()
                
            if st.button("MCP Servers", use_container_width=True):
                st.session_state.current_page = "MCP Servers"
                st.rerun()
            
            # Current cluster selection
            st.divider()
            st.subheader("Current Cluster")
            
            # Get clusters from database
            clusters = self.db_manager.get_all_clusters()
            cluster_names = ["None"] + [c["name"] for c in clusters]
            
            current_cluster = st.selectbox(
                "Select active cluster",
                options=cluster_names,
                index=0 if st.session_state.current_cluster is None else 
                      cluster_names.index(st.session_state.current_cluster),
            )
            
            if current_cluster == "None":
                st.session_state.current_cluster = None
            else:
                st.session_state.current_cluster = current_cluster
                
            # LLM provider selection
            st.divider()
            st.subheader("LLM Provider")
            
            # Get LLM configs from database
            llm_configs = self.db_manager.get_all_llm_configs()
            
            if llm_configs:
                # Create options in "provider/model" format
                llm_options = [
                    f"{c['provider']}/{c['model']}" for c in llm_configs
                ]
                
                # Find default config
                default_config = next(
                    (c for c in llm_configs if c["is_default"]), llm_configs[0]
                )
                default_option = f"{default_config['provider']}/{default_config['model']}"
                
                st.selectbox(
                    "LLM Provider/Model",
                    options=llm_options,
                    index=llm_options.index(default_option),
                )
            else:
                st.info("No LLM configurations found. Please add one in the API.")
    
    def _render_chat_page(self):
        """Render the chat page."""
        st.header("Chat")
        
        # Show current cluster context if any
        if st.session_state.current_cluster:
            st.info(f"Active cluster: {st.session_state.current_cluster}")
        
        # Display chat history
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.chat_history:
                role = message["role"]
                content = message["content"]
                
                if role == "user":
                    st.markdown(
                        f"""
                        <div class="chat-message user">
                            <div style="font-weight: bold;">You</div>
                            <div>{content}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="chat-message bot">
                            <div style="font-weight: bold;">KubeMindNexus</div>
                            <div>{content}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        
        # Show thinking indicator
        if st.session_state.thinking:
            st.info("Thinking...")
        
        # Chat input
        with st.form("chat_form", clear_on_submit=True):
            user_message = st.text_area("Type your message:", height=100)
            submit_col, stop_col = st.columns([5, 1])
            
            with submit_col:
                submit = st.form_submit_button("Send", use_container_width=True)
                
            with stop_col:
                if st.form_submit_button("Stop", use_container_width=True):
                    st.session_state.thinking = False
                    st.info("Stopping the current operation...")
        
        # Process message when submitted
        if submit and user_message and not st.session_state.thinking:
            # Add user message to chat history
            st.session_state.chat_history.append(
                {"role": "user", "content": user_message}
            )
            
            # Start thinking
            st.session_state.thinking = True
            st.rerun()
            
            # Process message in background thread
            thread = threading.Thread(
                target=self._process_chat_message,
                args=(user_message,),
            )
            add_script_run_ctx(thread)
            thread.start()
    
    def _process_chat_message(self, message: str):
        """Process chat message in background thread.
        
        Args:
            message: User message.
        """
        try:
            # Get current cluster ID if any
            cluster_id = None
            if st.session_state.current_cluster:
                cluster = self.db_manager.get_cluster_by_name(st.session_state.current_cluster)
                if cluster:
                    cluster_id = cluster["id"]
            
            # Get chat history for context
            chat_history = self.db_manager.get_chat_history(limit=10, cluster_id=cluster_id)
            conversation_history = [
                (msg["user_message"], msg["assistant_message"])
                for msg in reversed(chat_history)  # Reverse to get oldest messages first
            ]
            
            # Use asyncio.run instead of manually creating an event loop
            response, chat_id = asyncio.run(
                self.react_loop.run(
                    user_message=message,
                    conversation_history=conversation_history,
                    current_cluster=st.session_state.current_cluster,
                )
            )
            
            # Add assistant response to chat history
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response}
            )
                
        except Exception as e:
            # Add error message to chat history
            error_message = f"Error processing your message: {str(e)}"
            st.session_state.chat_history.append(
                {"role": "assistant", "content": error_message}
            )
            logger.error(f"Error in chat processing: {str(e)}")
            
        finally:
            # Stop thinking indicator
            st.session_state.thinking = False
            st.rerun()
    
    def _render_clusters_page(self):
        """Render the clusters management page."""
        st.header("Kubernetes Clusters")
        
        # Get all clusters
        clusters = self.db_manager.get_all_clusters()
        
        # Tabs for different cluster views
        list_tab, add_tab, metrics_tab = st.tabs(["Clusters List", "Add Cluster", "Cluster Metrics"])
        
        with list_tab:
            if not clusters:
                st.info("No clusters registered. Use the 'Add Cluster' tab to register one.")
            else:
                # Display clusters as a table
                clusters_df = pd.DataFrame(clusters)
                clusters_df = clusters_df[["id", "name", "ip", "port", "description", "created_at"]]
                
                st.dataframe(clusters_df, use_container_width=True)
                
                # Cluster details and actions
                st.subheader("Cluster Actions")
                
                # Select cluster for actions
                cluster_id = st.selectbox(
                    "Select cluster for actions",
                    options=[(c["id"], c["name"]) for c in clusters],
                    format_func=lambda x: f"{x[1]} (ID: {x[0]})",
                )
                
                if cluster_id:
                    selected_id, selected_name = cluster_id
                    
                    # Get cluster and its servers
                    cluster = self.db_manager.get_cluster(selected_id)
                    servers = self.db_manager.get_mcp_servers_by_cluster(selected_id)
                    
                    # Display cluster details
                    with st.expander("Cluster Details", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Name:** {cluster['name']}")
                            st.markdown(f"**IP Address:** {cluster['ip']}")
                            st.markdown(f"**Port:** {cluster['port']}")
                        with col2:
                            st.markdown(f"**ID:** {cluster['id']}")
                            st.markdown(f"**Created At:** {cluster['created_at']}")
                            st.markdown(f"**Description:** {cluster['description'] or 'N/A'}")
                    
                    # Display cluster MCP servers
                    with st.expander("Cluster MCP Servers", expanded=True):
                        if not servers:
                            st.info("No MCP servers associated with this cluster.")
                        else:
                            for server in servers:
                                st.markdown(f"**{server['name']}** ({server['type']})")
                                st.markdown(f"Status: {'Connected' if self.mcp_hub.manager.is_server_connected(server['name']) else 'Disconnected'}")
                                
                                # Connect/disconnect buttons
                                col1, col2 = st.columns(2)
                                with col1:
                                    if not self.mcp_hub.manager.is_server_connected(server['name']):
                                        if st.button(f"Connect to {server['name']}", key=f"connect_{server['id']}"):
                                            self._connect_to_server(server['name'])
                                            st.rerun()
                                with col2:
                                    if self.mcp_hub.manager.is_server_connected(server['name']):
                                        if st.button(f"Disconnect from {server['name']}", key=f"disconnect_{server['id']}"):
                                            self._disconnect_from_server(server['name'])
                                            st.rerun()
                                            
                                st.divider()
                    
                    # Cluster deletion
                    with st.expander("Danger Zone", expanded=False):
                        st.warning("Deleting a cluster will also disconnect and remove all associated MCP servers.")
                        
                        # Require confirmation to delete
                        confirm = st.text_input(
                            f"Type the cluster name '{selected_name}' to confirm deletion:",
                            key=f"delete_confirm_{selected_id}",
                        )
                        
                        if st.button("Delete Cluster", key=f"delete_{selected_id}"):
                            if confirm == selected_name:
                                self._delete_cluster(selected_id)
                                st.success(f"Cluster '{selected_name}' deleted successfully.")
                                st.rerun()
                            else:
                                st.error("Cluster name does not match. Deletion aborted.")
        
        with add_tab:
            st.subheader("Register New Cluster")
            
            with st.form("add_cluster_form"):
                name = st.text_input("Cluster Name")
                ip = st.text_input("IP Address")
                port = st.number_input("Port", min_value=1, max_value=65535, value=8080)
                description = st.text_area("Description (Optional)")
                
                if st.form_submit_button("Register Cluster"):
                    if name and ip:
                        self._add_cluster(name, ip, port, description)
                        st.success(f"Cluster '{name}' registered successfully.")
                        st.rerun()
                    else:
                        st.error("Cluster name and IP address are required.")
        
        with metrics_tab:
            if not clusters:
                st.info("No clusters registered. Use the 'Add Cluster' tab to register one.")
            else:
                # Select cluster for metrics
                cluster_id = st.selectbox(
                    "Select cluster for metrics",
                    options=[(c["id"], c["name"]) for c in clusters],
                    format_func=lambda x: f"{x[1]} (ID: {x[0]})",
                    key="metrics_cluster_select",
                )
                
                if cluster_id:
                    selected_id, selected_name = cluster_id
                    
                    # Example metrics tabs (these would be populated by actual data in a real implementation)
                    perf_tab, health_tab, storage_tab = st.tabs(["Performance", "Health", "Storage"])
                    
                    with perf_tab:
                        st.subheader(f"Performance Metrics for {selected_name}")
                        
                        # Example CPU usage chart
                        cpu_data = {
                            "time": [f"T-{i}" for i in range(10, 0, -1)],
                            "usage": [round(40 + 30 * (0.5 + 0.5 * (i % 3 - 1)), 2) for i in range(10)]
                        }
                        cpu_df = pd.DataFrame(cpu_data)
                        
                        st.line_chart(cpu_df.set_index("time"))
                        
                        # Example memory usage chart
                        memory_data = {
                            "time": [f"T-{i}" for i in range(10, 0, -1)],
                            "usage_gb": [round(4 + 2 * (0.5 + 0.5 * (i % 5 - 2)), 2) for i in range(10)]
                        }
                        memory_df = pd.DataFrame(memory_data)
                        
                        st.line_chart(memory_df.set_index("time"))
                        
                    with health_tab:
                        st.subheader(f"Health Status for {selected_name}")
                        
                        # Example health metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Node Status", "Healthy", "↑")
                            
                        with col2:
                            st.metric("Pod Health", "98%", "↑2%")
                            
                        with col3:
                            st.metric("Services", "15/15", "")
                            
                        # Example pod status chart
                        pod_status = {
                            "Status": ["Running", "Pending", "Failed", "Succeeded", "Unknown"],
                            "Count": [25, 2, 0, 5, 0]
                        }
                        pod_df = pd.DataFrame(pod_status)
                        
                        fig = px.pie(pod_df, values="Count", names="Status", title="Pod Status")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with storage_tab:
                        st.subheader(f"Storage Usage for {selected_name}")
                        
                        # Example storage chart
                        storage_data = {
                            "PV Name": [f"pv-{i}" for i in range(1, 6)],
                            "Capacity (GB)": [100, 50, 200, 75, 150],
                            "Used (GB)": [78, 32, 150, 60, 25]
                        }
                        storage_df = pd.DataFrame(storage_data)
                        
                        storage_df["Available (GB)"] = storage_df["Capacity (GB)"] - storage_df["Used (GB)"]
                        storage_df["Usage (%)"] = (storage_df["Used (GB)"] / storage_df["Capacity (GB)"] * 100).round(1)
                        
                        st.dataframe(storage_df, use_container_width=True)
                        
                        # Bar chart for storage usage
                        fig = px.bar(
                            storage_df,
                            x="PV Name",
                            y=["Used (GB)", "Available (GB)"],
                            title="Storage Usage",
                            barmode="stack"
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    def _render_mcp_servers_page(self):
        """Render the MCP servers management page."""
        st.header("MCP Servers")
        
        # Get all MCP servers
        servers = self.db_manager.get_all_mcp_servers()
        
        # Tabs for different MCP server views
        list_tab, add_tab, tools_tab = st.tabs(["Servers List", "Add Server", "Available Tools"])
        
        with list_tab:
            if not servers:
                st.info("No MCP servers registered. Use the 'Add Server' tab to register one.")
            else:
                # Display servers as a table
                st.subheader("All MCP Servers")
                
                # Create a simplified view for the table
                servers_view = []
                for server in servers:
                    cluster_name = "None"
                    if server["cluster_id"]:
                        cluster = self.db_manager.get_cluster(server["cluster_id"])
                        if cluster:
                            cluster_name = cluster["name"]
                            
                    servers_view.append({
                        "id": server["id"],
                        "name": server["name"],
                        "type": server["type"],
                        "cluster": cluster_name,
                        "is_connected": self.mcp_hub.manager.is_server_connected(server["name"]),
                        "is_local": bool(server["is_local"]),
                        "is_default": bool(server["is_default"])
                    })
                
                servers_df = pd.DataFrame(servers_view)
                st.dataframe(servers_df, use_container_width=True)
                
                # Server details and actions
                st.subheader("Server Actions")
                
                # Select server for actions
                server_id = st.selectbox(
                    "Select server for actions",
                    options=[(s["id"], s["name"]) for s in servers],
                    format_func=lambda x: f"{x[1]} (ID: {x[0]})",
                )
                
                if server_id:
                    selected_id, selected_name = server_id
                    
                    # Get server
                    server = self.db_manager.get_mcp_server(selected_id)
                    
                    # Display server details
                    with st.expander("Server Details", expanded=True):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Name:** {server['name']}")
                            st.markdown(f"**Type:** {server['type']}")
                            
                            if server["type"] == "stdio":
                                st.markdown(f"**Command:** {server['command']}")
                                if server["args"]:
                                    st.markdown(f"**Args:** {', '.join(server['args'])}")
                            else:
                                st.markdown(f"**URL:** {server['url']}")
                                
                        with col2:
                            cluster_name = "None"
                            if server["cluster_id"]:
                                cluster = self.db_manager.get_cluster(server["cluster_id"])
                                if cluster:
                                    cluster_name = cluster["name"]
                            
                            st.markdown(f"**Cluster:** {cluster_name}")
                            st.markdown(f"**Local:** {'Yes' if server['is_local'] else 'No'}")
                            st.markdown(f"**Default:** {'Yes' if server['is_default'] else 'No'}")
                            st.markdown(f"**Status:** {'Connected' if self.mcp_hub.manager.is_server_connected(server['name']) else 'Disconnected'}")
                    
                    # Connect/disconnect buttons
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if not self.mcp_hub.manager.is_server_connected(server['name']):
                            if st.button(f"Connect to {server['name']}", key=f"mcp_connect_{server['id']}"):
                                self._connect_to_server(server['name'])
                                st.success(f"Connected to server {server['name']}")
                                st.rerun()
                                
                    with col2:
                        if self.mcp_hub.manager.is_server_connected(server['name']):
                            if st.button(f"Disconnect from {server['name']}", key=f"mcp_disconnect_{server['id']}"):
                                self._disconnect_from_server(server['name'])
                                st.success(f"Disconnected from server {server['name']}")
                                st.rerun()
                    
                    # Server deletion
                    with st.expander("Danger Zone", expanded=False):
                        # Require confirmation to delete
                        confirm = st.text_input(
                            f"Type the server name '{selected_name}' to confirm deletion:",
                            key=f"mcp_delete_confirm_{selected_id}",
                        )
                        
                        if st.button("Delete Server", key=f"mcp_delete_{selected_id}"):
                            if confirm == selected_name:
                                self._delete_server(selected_id)
                                st.success(f"Server '{selected_name}' deleted successfully.")
                                st.rerun()
                            else:
                                st.error("Server name does not match. Deletion aborted.")
                    
        with add_tab:
            st.subheader("Add MCP Server")
            
            with st.form("add_server_form"):
                name = st.text_input("Server Name")
                server_type = st.selectbox("Server Type", ["stdio", "sse"])
                
                # Get all clusters for selection
                clusters = self.db_manager.get_all_clusters()
                cluster_options = [(None, "None")] + [(c["id"], c["name"]) for c in clusters]
                
                cluster_id = st.selectbox(
                    "Cluster",
                    options=cluster_options,
                    format_func=lambda x: x[1],
                )
                
                if server_type == "stdio":
                    command = st.text_input("Command")
                    args_str = st.text_input("Arguments (comma-separated)")
                    url = None
                else:
                    command = None
                    args_str = None
                    url = st.text_input("URL")
                
                is_local = st.checkbox("Local Server")
                is_default = st.checkbox("Default Server")
                
                env_str = st.text_area("Environment Variables (key=value, one per line)")
                
                if st.form_submit_button("Add Server"):
                    if name:
                        # Process arguments
                        args = None
                        if args_str:
                            args = [arg.strip() for arg in args_str.split(",")]
                            
                        # Process environment variables
                        env = {}
                        if env_str:
                            for line in env_str.strip().split("\n"):
                                if "=" in line:
                                    key, value = line.split("=", 1)
                                    env[key.strip()] = value.strip()
                        
                        # Extract selected cluster ID (if any)
                        selected_cluster_id = cluster_id[0] if cluster_id and cluster_id[0] is not None else None
                        
                        try:
                            # Create server
                            server_id = asyncio.run(self.mcp_hub.manager.add_server(
                                name=name,
                                server_type=server_type,
                                command=command,
                                args=args,
                                url=url,
                                cluster_id=selected_cluster_id,
                                is_local=is_local,
                                is_default=is_default,
                                env=env,
                            ))
                            
                            st.success(f"Server '{name}' added successfully.")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Failed to add server: {str(e)}")
                    else:
                        st.error("Server name is required.")
                        
        with tools_tab:
            # Get all connected servers
            connected_servers = self.mcp_hub.manager.get_connected_servers()
            
            if not connected_servers:
                st.info("No connected MCP servers. Connect to a server to view available tools.")
            else:
                # Get all available tools
                all_tools = self.mcp_hub.get_all_available_tools()
                
                if not all_tools:
                    st.info("No tools available from connected servers.")
                else:
                    for server_name, tools in all_tools.items():
                        st.subheader(f"Tools from {server_name}")
                        
                        if not tools:
                            st.info(f"No tools available from {server_name}.")
                            continue
                            
                        # Display tools in an expander
                        for tool in tools:
                            tool_name = tool.get("name", "Unknown")
                            tool_desc = tool.get("description", "No description available")
                            
                            with st.expander(f"{tool_name}", expanded=False):
                                st.markdown(f"**Description:** {tool_desc}")
                                
                                # Display input schema if available
                                if "inputSchema" in tool and "properties" in tool["inputSchema"]:
                                    st.markdown("**Parameters:**")
                                    
                                    properties = tool["inputSchema"]["properties"]
                                    required = tool["inputSchema"].get("required", [])
                                    
                                    for param_name, param_info in properties.items():
                                        param_type = param_info.get("type", "any")
                                        param_desc = param_info.get("description", "")
                                        is_required = param_name in required
                                        req_marker = " (required)" if is_required else " (optional)"
                                        
                                        st.markdown(f"- **{param_name}**: {param_type}{req_marker} - {param_desc}")
    
    # Helper methods
    def _connect_to_server(self, server_name: str) -> bool:
        """Connect to an MCP server.
        
        Args:
            server_name: Server name.
            
        Returns:
            True if connection was successful, False otherwise.
        """
        try:
            # Use asyncio.run instead of manually creating an event loop
            success = asyncio.run(self.mcp_hub.manager.connect_server(server_name))
            return success
        except Exception as e:
            logger.error(f"Error connecting to server {server_name}: {str(e)}")
            return False
    
    def _disconnect_from_server(self, server_name: str) -> bool:
        """Disconnect from an MCP server.
        
        Args:
            server_name: Server name.
            
        Returns:
            True if disconnection was successful, False otherwise.
        """
        try:
            # Use asyncio.run instead of manually creating an event loop
            success = asyncio.run(self.mcp_hub.manager.disconnect_server(server_name))
            return success
        except Exception as e:
            logger.error(f"Error disconnecting from server {server_name}: {str(e)}")
            return False
    
    def _add_cluster(self, name: str, ip: str, port: int, description: Optional[str] = None) -> int:
        """Add a new cluster.
        
        Args:
            name: Cluster name.
            ip: Cluster IP address.
            port: Cluster port.
            description: Optional cluster description.
            
        Returns:
            Cluster ID.
        """
        try:
            # Add cluster to database
            cluster_id = self.db_manager.add_cluster(
                name=name,
                ip=ip,
                port=port,
                description=description,
            )
            
            return cluster_id
            
        except Exception as e:
            logger.error(f"Error adding cluster: {str(e)}")
            raise
    
    def _delete_cluster(self, cluster_id: int) -> bool:
        """Delete a cluster.
        
        Args:
            cluster_id: Cluster ID.
            
        Returns:
            True if deletion was successful, False otherwise.
        """
        try:
            # Get MCP servers for this cluster
            servers = self.db_manager.get_mcp_servers_by_cluster(cluster_id)
            
            # Disconnect any connected servers
            for server in servers:
                self._disconnect_from_server(server["name"])
                
            # Delete cluster
            success = self.db_manager.delete_cluster(cluster_id)
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting cluster: {str(e)}")
            return False
    
    def _delete_server(self, server_id: int) -> bool:
        """Delete an MCP server.
        
        Args:
            server_id: Server ID.
            
        Returns:
            True if deletion was successful, False otherwise.
        """
        try:
            # Get server
            server = self.db_manager.get_mcp_server(server_id)
            
            if not server:
                return False
                
            # Disconnect if connected
            if self.mcp_hub.manager.is_server_connected(server["name"]):
                self._disconnect_from_server(server["name"])
                
            # Use asyncio.run instead of manually creating an event loop
            success = asyncio.run(self.mcp_hub.manager.remove_server(server_id))
            return success
                
        except Exception as e:
            logger.error(f"Error deleting server: {str(e)}")
            return False


def run_app(config: Configuration, db_manager: DatabaseManager, mcp_hub: MCPHub, react_loop: ReactLoop):
    """Run the Streamlit application.
    
    Args:
        config: Configuration instance.
        db_manager: Database manager instance.
        mcp_hub: MCP hub instance.
        react_loop: ReactLoop instance.
    """
    app = StreamlitApp(config, db_manager, mcp_hub, react_loop)
    app.run()
