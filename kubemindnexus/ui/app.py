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

from .api_client import ApiClient

logger = logging.getLogger(__name__)


class AsyncToSync:
    """Helper class to run async functions in a synchronous context."""
    
    _loop = None
    
    @classmethod
    def run(cls, coro):
        """Run an async coroutine synchronously.
        
        Args:
            coro: Coroutine to run.
            
        Returns:
            Result of the coroutine.
        """
        if cls._loop is None or cls._loop.is_closed():
            cls._loop = asyncio.new_event_loop()
            
        try:
            return cls._loop.run_until_complete(coro)
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                # Create a new loop if the previous one was closed
                cls._loop = asyncio.new_event_loop()
                return cls._loop.run_until_complete(coro)
            raise


class StreamlitApp:
    """Streamlit UI application for KubeMindNexus."""
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
    ):
        """Initialize Streamlit application.
        
        Args:
            api_url: URL of the API server.
        """
        # Initialize API client
        self.api_client = ApiClient(base_url=api_url)
        
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
            
        if "running" not in st.session_state:
            st.session_state.running = True
    
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
        
        # Close API client when app is closed
        if not st.session_state.running:
            AsyncToSync.run(self.api_client.close())
    
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
            
            # Get clusters from API
            clusters = AsyncToSync.run(self.api_client.get_clusters())
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
            
            # Get LLM configs from API
            llm_configs = AsyncToSync.run(self.api_client.get_llm_configs())
            
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
                cluster = AsyncToSync.run(
                    self.api_client.get_cluster_by_name(st.session_state.current_cluster)
                )
                if cluster:
                    cluster_id = cluster["id"]
            
            # Send message to API
            response = AsyncToSync.run(
                self.api_client.send_chat_message(
                    message=message,
                    cluster_id=cluster_id,
                )
            )
            
            if response:
                # Add assistant response to chat history
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response["message"]}
                )
            else:
                # Add error message to chat history
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": "Failed to process your message. Please try again."}
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
        
        # Get all clusters from API
        clusters = AsyncToSync.run(self.api_client.get_clusters())
        
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
                    
                    # Get cluster and its servers from API
                    cluster = AsyncToSync.run(self.api_client.get_cluster(selected_id))
                    servers = AsyncToSync.run(self.api_client.get_mcp_servers_by_cluster(selected_id))
                    
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
                                
                                # Get server status from API
                                server_status = AsyncToSync.run(
                                    self.api_client.get_mcp_server_status(server['id'])
                                )
                                is_connected = server_status["is_connected"] if server_status else False
                                status_text = "Connected" if is_connected else "Disconnected"
                                status_color = "green" if is_connected else "red"
                                
                                st.markdown(f"Status: <span style='color:{status_color}'>{status_text}</span>", unsafe_allow_html=True)
                                
                                # Connect/disconnect buttons
                                col1, col2 = st.columns(2)
                                with col1:
                                    if not is_connected:
                                        if st.button(f"Connect to {server['name']}", key=f"connect_{server['id']}"):
                                            success = AsyncToSync.run(
                                                self.api_client.connect_mcp_server(server['id'])
                                            )
                                            if success:
                                                st.success(f"Connected to server {server['name']}")
                                                st.rerun()
                                            else:
                                                st.error(f"Failed to connect to server {server['name']}")
                                with col2:
                                    if is_connected:
                                        if st.button(f"Disconnect from {server['name']}", key=f"disconnect_{server['id']}"):
                                            success = AsyncToSync.run(
                                                self.api_client.disconnect_mcp_server(server['id'])
                                            )
                                            if success:
                                                st.success(f"Disconnected from server {server['name']}")
                                                st.rerun()
                                            else:
                                                st.error(f"Failed to disconnect from server {server['name']}")
                                            
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
                                success = AsyncToSync.run(
                                    self.api_client.delete_cluster(selected_id)
                                )
                                if success:
                                    st.success(f"Cluster '{selected_name}' deleted successfully.")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to delete cluster '{selected_name}'.")
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
                        cluster_id = AsyncToSync.run(
                            self.api_client.add_cluster(
                                name=name,
                                ip=ip,
                                port=port,
                                description=description,
                            )
                        )
                        if cluster_id:
                            st.success(f"Cluster '{name}' registered successfully.")
                            st.rerun()
                        else:
                            st.error("Failed to register cluster.")
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
                    
                    # Metrics tabs
                    perf_tab, health_tab, storage_tab, resources_tab = st.tabs(["Performance", "Health", "Storage", "Resources"])
                    
                    with perf_tab:
                        st.subheader(f"Performance Metrics for {selected_name}")
                        
                        # Loading spinner
                        with st.spinner("Loading performance metrics..."):
                            # Get performance metrics from API
                            metrics = AsyncToSync.run(
                                self.api_client.get_cluster_performance_metrics(selected_id)
                            )
                            
                            if metrics:
                                # CPU usage chart
                                cpu_data = pd.DataFrame(metrics["cpu_usage"])
                                st.caption("CPU Usage Over Time")
                                st.line_chart(cpu_data.set_index("time"))
                                
                                # Memory usage chart
                                memory_data = pd.DataFrame(metrics["memory_usage"])
                                st.caption("Memory Usage Over Time")
                                st.line_chart(memory_data.set_index("time"))
                            else:
                                st.info("No performance metrics available for this cluster.")
                        
                    with health_tab:
                        st.subheader(f"Health Status for {selected_name}")
                        
                        # Loading spinner
                        with st.spinner("Loading health metrics..."):
                            # Get health metrics from API
                            metrics = AsyncToSync.run(
                                self.api_client.get_cluster_health_metrics(selected_id)
                            )
                            
                            if metrics:
                                # Health metrics
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Node Status", metrics["node_status"])
                                    
                                with col2:
                                    st.metric("Pod Health", f"{metrics['pod_health_percentage']}%")
                                    
                                with col3:
                                    services_count = metrics["services_count"]
                                    st.metric(
                                        "Services", 
                                        f"{services_count.get('Healthy', 0)}/{services_count.get('Total', 0)}"
                                    )
                                    
                                # Pod status chart
                                pod_status = metrics["pod_status"]
                                pod_df = pd.DataFrame({
                                    "Status": list(pod_status.keys()),
                                    "Count": list(pod_status.values())
                                })
                                
                                fig = px.pie(
                                    pod_df, 
                                    values="Count", 
                                    names="Status", 
                                    title="Pod Status"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No health metrics available for this cluster.")
                        
                    with storage_tab:
                        st.subheader(f"Storage Usage for {selected_name}")
                        
                        # Loading spinner
                        with st.spinner("Loading storage metrics..."):
                            # Get storage metrics from API
                            metrics = AsyncToSync.run(
                                self.api_client.get_cluster_storage_metrics(selected_id)
                            )
                            
                            if metrics:
                                # Storage chart
                                storage_data = metrics["storage_usage"]
                                storage_df = pd.DataFrame([
                                    {
                                        "PV Name": item["pv_name"],
                                        "Capacity (GB)": item["capacity_gb"],
                                        "Used (GB)": item["used_gb"],
                                        "Available (GB)": item["available_gb"],
                                        "Usage (%)": item["usage_percentage"]
                                    }
                                    for item in storage_data
                                ])
                                
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
                            else:
                                st.info("No storage metrics available for this cluster.")
                                
                    with resources_tab:
                        st.subheader(f"Kubernetes Resources for {selected_name}")
                        
                        resources_tabs = st.tabs(["Nodes", "Pods", "Services", "Persistent Volumes"])
                        
                        with resources_tabs[0]:  # Nodes tab
                            with st.spinner("Loading nodes..."):
                                nodes = AsyncToSync.run(
                                    self.api_client.get_cluster_nodes(selected_id)
                                )
                                
                                if nodes:
                                    nodes_df = pd.DataFrame(nodes)
                                    st.dataframe(nodes_df, use_container_width=True)
                                else:
                                    st.info("No node information available for this cluster.")
                                    
                        with resources_tabs[1]:  # Pods tab
                            namespace = st.selectbox(
                                "Namespace",
                                ["All Namespaces", "default", "kube-system", "monitoring"],
                                key="pods_namespace"
                            )
                            
                            with st.spinner("Loading pods..."):
                                ns = None if namespace == "All Namespaces" else namespace
                                pods = AsyncToSync.run(
                                    self.api_client.get_cluster_pods(selected_id, namespace=ns)
                                )
                                
                                if pods:
                                    pods_df = pd.DataFrame(pods)
                                    st.dataframe(pods_df, use_container_width=True)
                                else:
                                    st.info("No pod information available for this cluster.")
                                    
                        with resources_tabs[2]:  # Services tab
                            namespace = st.selectbox(
                                "Namespace",
                                ["All Namespaces", "default", "kube-system", "monitoring"],
                                key="services_namespace"
                            )
                            
                            with st.spinner("Loading services..."):
                                ns = None if namespace == "All Namespaces" else namespace
                                services = AsyncToSync.run(
                                    self.api_client.get_cluster_services(selected_id, namespace=ns)
                                )
                                
                                if services:
                                    services_df = pd.DataFrame(services)
                                    st.dataframe(services_df, use_container_width=True)
                                else:
                                    st.info("No service information available for this cluster.")
                                    
                        with resources_tabs[3]:  # Persistent Volumes tab
                            with st.spinner("Loading persistent volumes..."):
                                pvs = AsyncToSync.run(
                                    self.api_client.get_cluster_persistent_volumes(selected_id)
                                )
                                
                                if pvs:
                                    pvs_df = pd.DataFrame(pvs)
                                    st.dataframe(pvs_df, use_container_width=True)
                                else:
                                    st.info("No persistent volume information available for this cluster.")
    
    def _render_mcp_servers_page(self):
        """Render the MCP servers management page."""
        st.header("MCP Servers")
        
        # Get all MCP servers from API
        servers = AsyncToSync.run(self.api_client.get_mcp_servers())
        
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
                # Get server status
                servers_status = AsyncToSync.run(self.api_client.get_mcp_servers_status())
                status_dict = {s["id"]: s["is_connected"] for s in servers_status}
                
                for server in servers:
                    cluster_name = "None"
                    if server["cluster_id"]:
                        cluster = AsyncToSync.run(self.api_client.get_cluster(server["cluster_id"]))
                        if cluster:
                            cluster_name = cluster["name"]
                    
                    # Get connection status from status endpoint
                    is_connected = status_dict.get(server["id"], False)
                    
                    servers_view.append({
                        "id": server["id"],
                        "name": server["name"],
                        "type": server["type"],
                        "cluster": cluster_name,
                        "is_connected": is_connected,
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
                    server = AsyncToSync.run(self.api_client.get_mcp_server(selected_id))
                    
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
                                cluster = AsyncToSync.run(self.api_client.get_cluster(server["cluster_id"]))
                                if cluster:
                                    cluster_name = cluster["name"]
                            
                            # Get server status from API
                            server_status = AsyncToSync.run(
                                self.api_client.get_mcp_server_status(selected_id)
                            )
                            is_connected = server_status["is_connected"] if server_status else False
                            status_text = "Connected" if is_connected else "Disconnected"
                            status_color = "green" if is_connected else "red"
                            
                            st.markdown(f"**Cluster:** {cluster_name}")
                            st.markdown(f"**Local:** {'Yes' if server['is_local'] else 'No'}")
                            st.markdown(f"**Default:** {'Yes' if server['is_default'] else 'No'}")
                            st.markdown(f"**Status:** <span style='color:{status_color}'>{status_text}</span>", unsafe_allow_html=True)
                    
                    # Connect/disconnect buttons
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button(f"Connect to {server['name']}", key=f"mcp_connect_{server['id']}"):
                            success = AsyncToSync.run(
                                self.api_client.connect_mcp_server(server["id"])
                            )
                            if success:
                                st.success(f"Connected to server {server['name']}")
                                st.rerun()
                            else:
                                st.error(f"Failed to connect to server {server['name']}")
                                
                    with col2:
                        if st.button(f"Disconnect from {server['name']}", key=f"mcp_disconnect_{server['id']}"):
                            success = AsyncToSync.run(
                                self.api_client.disconnect_mcp_server(server["id"])
                            )
                            if success:
                                st.success(f"Disconnected from server {server['name']}")
                                st.rerun()
                            else:
                                st.error(f"Failed to disconnect from server {server['name']}")
                    
                    # Server deletion
                    with st.expander("Danger Zone", expanded=False):
                        # Require confirmation to delete
                        confirm = st.text_input(
                            f"Type the server name '{selected_name}' to confirm deletion:",
                            key=f"mcp_delete_confirm_{selected_id}",
                        )
                        
                        if st.button("Delete Server", key=f"mcp_delete_{selected_id}"):
                            if confirm == selected_name:
                                success = AsyncToSync.run(
                                    self.api_client.delete_mcp_server(selected_id)
                                )
                                if success:
                                    st.success(f"Server '{selected_name}' deleted successfully.")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to delete server '{selected_name}'.")
                            else:
                                st.error("Server name does not match. Deletion aborted.")
                    
        with add_tab:
            st.subheader("Add MCP Server")
            
            with st.form("add_server_form"):
                name = st.text_input("Server Name")
                server_type = st.selectbox("Server Type", ["stdio", "sse"])
                
                # Get all clusters for selection
                clusters = AsyncToSync.run(self.api_client.get_clusters())
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
                            # Create server through API
                            server_id = AsyncToSync.run(
                                self.api_client.add_mcp_server(
                                    name=name,
                                    server_type=server_type,
                                    command=command,
                                    args=args,
                                    url=url,
                                    cluster_id=selected_cluster_id,
                                    is_local=is_local,
                                    is_default=is_default,
                                    env=env,
                                )
                            )
                            
                            if server_id:
                                st.success(f"Server '{name}' added successfully.")
                                st.rerun()
                            else:
                                st.error("Failed to add server.")
                            
                        except Exception as e:
                            st.error(f"Failed to add server: {str(e)}")
                    else:
                        st.error("Server name is required.")
                        
        with tools_tab:
            if not servers:
                st.info("No MCP servers registered. Use the 'Add Server' tab to register one.")
            else:
                # Select server for tools
                server_id = st.selectbox(
                    "Select server to view tools",
                    options=[(s["id"], s["name"]) for s in servers],
                    format_func=lambda x: f"{x[1]} (ID: {x[0]})",
                    key="tools_server_select",
                )
                
                if server_id:
                    selected_id, selected_name = server_id
                    
                    # Get server status
                    server_status = AsyncToSync.run(
                        self.api_client.get_mcp_server_status(selected_id)
                    )
                    is_connected = server_status["is_connected"] if server_status else False
                    
                    if not is_connected:
                        st.warning(f"Server '{selected_name}' is not connected. Connect to the server to view available tools.")
                        
                        if st.button("Connect Server", key="tools_connect_server"):
                            success = AsyncToSync.run(
                                self.api_client.connect_mcp_server(selected_id)
                            )
                            if success:
                                st.success(f"Connected to server {selected_name}")
                                st.rerun()
                            else:
                                st.error(f"Failed to connect to server {selected_name}")
                    else:
                        # Fetch tools and resources
                        tools_tab, resources_tab = st.tabs(["Tools", "Resources"])
                        
                        with tools_tab:
                            with st.spinner("Loading tools..."):
                                tools = AsyncToSync.run(
                                    self.api_client.get_mcp_server_tools(selected_id)
                                )
                                
                                if tools:
                                    for tool in tools:
                                        with st.expander(f"Tool: {tool.get('name', 'Unknown')}"):
                                            st.json(tool)
                                else:
                                    st.info(f"No tools available for server '{selected_name}'")
                        
                        with resources_tab:
                            with st.spinner("Loading resources..."):
                                resources = AsyncToSync.run(
                                    self.api_client.get_mcp_server_resources(selected_id)
                                )
                                
                                if resources:
                                    for resource in resources:
                                        with st.expander(f"Resource: {resource.get('uri', 'Unknown')}"):
                                            st.json(resource)
                                else:
                                    st.info(f"No resources available for server '{selected_name}'")


def run_app(api_url: str = "http://localhost:8000"):
    """Run the Streamlit application.
    
    Args:
        api_url: URL of the API server.
    """
    app = StreamlitApp(api_url=api_url)
    app.run()
