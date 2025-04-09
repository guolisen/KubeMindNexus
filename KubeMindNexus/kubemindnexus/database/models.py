"""Database models for KubeMindNexus."""
import datetime
import json
from typing import Dict, Any, Optional, List, Union

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Cluster(Base):
    """Database model for a Kubernetes cluster."""
    
    __tablename__ = "clusters"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    ip = Column(String(255), nullable=False)
    port = Column(Integer, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Relationships
    servers = relationship("MCPServer", back_populates="cluster", cascade="all, delete-orphan")
    chat_history = relationship("ChatHistory", back_populates="cluster", cascade="all, delete-orphan")
    health_checks = relationship("HealthCheck", back_populates="cluster", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Cluster(name='{self.name}', ip='{self.ip}', port={self.port})>"


class MCPServer(Base):
    """Database model for an MCP server configuration."""
    
    __tablename__ = "mcp_servers"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    type = Column(String(50), nullable=False)  # 'stdio' or 'sse'
    command = Column(String(255), nullable=True)
    args = Column(JSON, nullable=True)
    env = Column(JSON, nullable=True)
    url = Column(String(255), nullable=True)
    is_local = Column(Boolean, default=False)
    cluster_id = Column(Integer, ForeignKey("clusters.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Relationships
    cluster = relationship("Cluster", back_populates="servers")
    
    def __repr__(self) -> str:
        return f"<MCPServer(name='{self.name}', type='{self.type}')>"
    
    @property
    def args_list(self) -> List[str]:
        """Get args as a list.
        
        Returns:
            List of arguments.
        """
        if self.args is None:
            return []
        
        if isinstance(self.args, list):
            return self.args
        
        if isinstance(self.args, str):
            try:
                return json.loads(self.args)
            except Exception:
                return []
        
        return []
    
    @property
    def env_dict(self) -> Dict[str, str]:
        """Get env as a dictionary.
        
        Returns:
            Dictionary of environment variables.
        """
        if self.env is None:
            return {}
        
        if isinstance(self.env, dict):
            return self.env
        
        if isinstance(self.env, str):
            try:
                return json.loads(self.env)
            except Exception:
                return {}
        
        return {}


class ChatHistory(Base):
    """Database model for chat history."""
    
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True)
    cluster_id = Column(Integer, ForeignKey("clusters.id"), nullable=True)
    user_message = Column(Text, nullable=False)
    assistant_response = Column(Text, nullable=False)
    llm_provider = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    chat_metadata = Column(JSON, nullable=True)  # Renamed from 'metadata' as it's a reserved keyword
    
    # Relationships
    cluster = relationship("Cluster", back_populates="chat_history")
    
    def __repr__(self) -> str:
        return f"<ChatHistory(id={self.id}, timestamp={self.timestamp})>"


class HealthCheck(Base):
    """Database model for cluster health checks."""
    
    __tablename__ = "health_checks"
    
    id = Column(Integer, primary_key=True)
    cluster_id = Column(Integer, ForeignKey("clusters.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String(50), nullable=False)  # 'healthy', 'warning', 'error'
    details = Column(JSON, nullable=True)
    
    # Relationships
    cluster = relationship("Cluster", back_populates="health_checks")
    
    def __repr__(self) -> str:
        return f"<HealthCheck(cluster='{self.cluster.name if self.cluster else None}', status='{self.status}')>"


class PerformanceMetric(Base):
    """Database model for cluster performance metrics."""
    
    __tablename__ = "performance_metrics"
    
    id = Column(Integer, primary_key=True)
    cluster_id = Column(Integer, ForeignKey("clusters.id"), nullable=False)
    metric_type = Column(String(50), nullable=False)  # 'cpu', 'memory', 'network', etc.
    value = Column(Float, nullable=False)
    unit = Column(String(20), nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    metric_metadata = Column(JSON, nullable=True)  # Renamed from 'metadata' as it's a reserved keyword
    
    # Relationships
    cluster = relationship("Cluster")
    
    def __repr__(self) -> str:
        return f"<PerformanceMetric(cluster='{self.cluster.name if self.cluster else None}', type='{self.metric_type}', value={self.value})>"


class Config(Base):
    """Database model for application configuration key-value pairs."""
    
    __tablename__ = "config"
    
    key = Column(String(255), primary_key=True)
    value = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    
    def __repr__(self) -> str:
        return f"<Config(key='{self.key}')>"
