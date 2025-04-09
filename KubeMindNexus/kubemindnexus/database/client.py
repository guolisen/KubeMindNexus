"""Database client for KubeMindNexus."""
import datetime
import logging
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Generic

from sqlalchemy import create_engine, exc
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.future import select

from ..config.settings import SQLITE_DB_PATH
from ..utils.logger import LoggerMixin
from .models import Base, Cluster, MCPServer, ChatHistory, HealthCheck, PerformanceMetric, Config

# Type variable for generic database model
T = TypeVar('T')


class DatabaseClient(LoggerMixin):
    """Database client for SQLite database operations."""
    
    def __init__(self, db_path: Optional[str] = None) -> None:
        """Initialize the database client.
        
        Args:
            db_path: Path to the SQLite database file. If None, uses the default path.
        """
        self.db_path = db_path or SQLITE_DB_PATH
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.Session = sessionmaker(bind=self.engine)
        
        # Create tables if they don't exist
        self._create_tables()
    
    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        try:
            Base.metadata.create_all(self.engine)
            self.logger.info(f"Database tables created at {self.db_path}")
        except Exception as e:
            self.logger.error(f"Error creating database tables: {e}")
    
    def get_session(self) -> Session:
        """Get a database session.
        
        Returns:
            A SQLAlchemy session.
        """
        return self.Session()
    
    def add(self, obj: Any) -> bool:
        """Add an object to the database.
        
        Args:
            obj: The object to add.
            
        Returns:
            True if the object was added successfully, False otherwise.
        """
        try:
            with self.get_session() as session:
                session.add(obj)
                session.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error adding object to database: {e}")
            return False
    
    def add_all(self, objects: List[Any]) -> bool:
        """Add multiple objects to the database.
        
        Args:
            objects: The objects to add.
            
        Returns:
            True if the objects were added successfully, False otherwise.
        """
        try:
            with self.get_session() as session:
                session.add_all(objects)
                session.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error adding objects to database: {e}")
            return False
    
    def update(self, obj: Any) -> bool:
        """Update an object in the database.
        
        Args:
            obj: The object to update.
            
        Returns:
            True if the object was updated successfully, False otherwise.
        """
        try:
            with self.get_session() as session:
                session.merge(obj)
                session.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error updating object in database: {e}")
            return False
    
    def delete(self, obj: Any) -> bool:
        """Delete an object from the database.
        
        Args:
            obj: The object to delete.
            
        Returns:
            True if the object was deleted successfully, False otherwise.
        """
        try:
            with self.get_session() as session:
                session.delete(obj)
                session.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error deleting object from database: {e}")
            return False
    
    def get(self, model: Type[T], id: Any) -> Optional[T]:
        """Get an object by ID.
        
        Args:
            model: The model class.
            id: The object ID.
            
        Returns:
            The object if found, None otherwise.
        """
        try:
            with self.get_session() as session:
                return session.query(model).get(id)
        except Exception as e:
            self.logger.error(f"Error getting object from database: {e}")
            return None
    
    def get_all(self, model: Type[T]) -> List[T]:
        """Get all objects of a model.
        
        Args:
            model: The model class.
            
        Returns:
            List of objects.
        """
        try:
            with self.get_session() as session:
                return session.query(model).all()
        except Exception as e:
            self.logger.error(f"Error getting objects from database: {e}")
            return []
    
    def query(self, model: Type[T], **kwargs) -> List[T]:
        """Query objects by attributes.
        
        Args:
            model: The model class.
            **kwargs: Query parameters.
            
        Returns:
            List of matching objects.
        """
        try:
            with self.get_session() as session:
                return session.query(model).filter_by(**kwargs).all()
        except Exception as e:
            self.logger.error(f"Error querying database: {e}")
            return []
    
    # Cluster-specific methods
    
    def add_cluster(self, name: str, ip: str, port: int, description: Optional[str] = None) -> Optional[Cluster]:
        """Add a new cluster to the database.
        
        Args:
            name: Name of the cluster.
            ip: IP address of the cluster.
            port: Port of the cluster.
            description: Optional description of the cluster.
            
        Returns:
            The added cluster if successful, None otherwise.
        """
        try:
            cluster = Cluster(
                name=name,
                ip=ip,
                port=port,
                description=description,
            )
            with self.get_session() as session:
                session.add(cluster)
                session.commit()
                session.refresh(cluster)
            return cluster
        except exc.IntegrityError:
            self.logger.error(f"Cluster with name '{name}' already exists")
            return None
        except Exception as e:
            self.logger.error(f"Error adding cluster to database: {e}")
            return None
    
    def get_cluster_by_name(self, name: str) -> Optional[Cluster]:
        """Get a cluster by name.
        
        Args:
            name: Name of the cluster.
            
        Returns:
            The cluster if found, None otherwise.
        """
        try:
            with self.get_session() as session:
                return session.query(Cluster).filter_by(name=name).first()
        except Exception as e:
            self.logger.error(f"Error getting cluster from database: {e}")
            return None
    
    def update_cluster(self, cluster_id: int, **kwargs) -> bool:
        """Update a cluster.
        
        Args:
            cluster_id: ID of the cluster to update.
            **kwargs: Attributes to update.
            
        Returns:
            True if the cluster was updated successfully, False otherwise.
        """
        try:
            with self.get_session() as session:
                cluster = session.query(Cluster).get(cluster_id)
                if not cluster:
                    self.logger.warning(f"Cluster with ID {cluster_id} not found")
                    return False
                
                for key, value in kwargs.items():
                    if hasattr(cluster, key):
                        setattr(cluster, key, value)
                
                cluster.updated_at = datetime.datetime.utcnow()
                session.commit()
                return True
        except Exception as e:
            self.logger.error(f"Error updating cluster in database: {e}")
            return False
    
    def delete_cluster(self, cluster_id: int) -> bool:
        """Delete a cluster.
        
        Args:
            cluster_id: ID of the cluster to delete.
            
        Returns:
            True if the cluster was deleted successfully, False otherwise.
        """
        try:
            with self.get_session() as session:
                cluster = session.query(Cluster).get(cluster_id)
                if not cluster:
                    self.logger.warning(f"Cluster with ID {cluster_id} not found")
                    return False
                
                session.delete(cluster)
                session.commit()
                return True
        except Exception as e:
            self.logger.error(f"Error deleting cluster from database: {e}")
            return False
    
    # MCP server methods
    
    def add_server(self, name: str, server_type: str, cluster_id: Optional[int] = None,
                  command: Optional[str] = None, args: Optional[List[str]] = None,
                  env: Optional[Dict[str, str]] = None, url: Optional[str] = None,
                  is_local: bool = False) -> Optional[MCPServer]:
        """Add a new MCP server to the database.
        
        Args:
            name: Name of the server.
            server_type: Type of the server ('stdio' or 'sse').
            cluster_id: ID of the cluster the server belongs to, or None for local servers.
            command: Command to run for stdio servers.
            args: Arguments for the command.
            env: Environment variables.
            url: URL for SSE servers.
            is_local: Whether the server is a local server.
            
        Returns:
            The added server if successful, None otherwise.
        """
        try:
            server = MCPServer(
                name=name,
                type=server_type,
                cluster_id=cluster_id,
                command=command,
                args=args or [],
                env=env or {},
                url=url,
                is_local=is_local,
            )
            with self.get_session() as session:
                session.add(server)
                session.commit()
                session.refresh(server)
            return server
        except Exception as e:
            self.logger.error(f"Error adding MCP server to database: {e}")
            return None
    
    def get_servers_for_cluster(self, cluster_id: int) -> List[MCPServer]:
        """Get all MCP servers for a cluster.
        
        Args:
            cluster_id: ID of the cluster.
            
        Returns:
            List of MCP servers.
        """
        try:
            with self.get_session() as session:
                return session.query(MCPServer).filter_by(cluster_id=cluster_id).all()
        except Exception as e:
            self.logger.error(f"Error getting MCP servers from database: {e}")
            return []
    
    def get_local_servers(self) -> List[MCPServer]:
        """Get all local MCP servers.
        
        Returns:
            List of local MCP servers.
        """
        try:
            with self.get_session() as session:
                return session.query(MCPServer).filter_by(is_local=True).all()
        except Exception as e:
            self.logger.error(f"Error getting local MCP servers from database: {e}")
            return []
    
    # Chat history methods
    
    def add_chat_history(self, user_message: str, assistant_response: str,
                        llm_provider: str, cluster_id: Optional[int] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> Optional[ChatHistory]:
        """Add a chat history entry to the database.
        
        Args:
            user_message: The user's message.
            assistant_response: The assistant's response.
            llm_provider: The LLM provider used.
            cluster_id: ID of the cluster the chat is about, or None if not specific.
            metadata: Optional metadata about the chat.
            
        Returns:
            The added chat history entry if successful, None otherwise.
        """
        try:
            chat = ChatHistory(
                user_message=user_message,
                assistant_response=assistant_response,
                llm_provider=llm_provider,
                cluster_id=cluster_id,
                metadata=metadata or {},
            )
            with self.get_session() as session:
                session.add(chat)
                session.commit()
                session.refresh(chat)
            return chat
        except Exception as e:
            self.logger.error(f"Error adding chat history to database: {e}")
            return None
    
    def get_chat_history(self, limit: int = 100, cluster_id: Optional[int] = None) -> List[ChatHistory]:
        """Get chat history.
        
        Args:
            limit: Maximum number of entries to return.
            cluster_id: ID of the cluster to filter by, or None for all.
            
        Returns:
            List of chat history entries.
        """
        try:
            with self.get_session() as session:
                query = session.query(ChatHistory)
                if cluster_id is not None:
                    query = query.filter_by(cluster_id=cluster_id)
                return query.order_by(ChatHistory.timestamp.desc()).limit(limit).all()
        except Exception as e:
            self.logger.error(f"Error getting chat history from database: {e}")
            return []
    
    # Health check methods
    
    def add_health_check(self, cluster_id: int, status: str,
                        details: Optional[Dict[str, Any]] = None) -> Optional[HealthCheck]:
        """Add a health check entry to the database.
        
        Args:
            cluster_id: ID of the cluster.
            status: Status of the cluster ('healthy', 'warning', 'error').
            details: Optional details about the health check.
            
        Returns:
            The added health check entry if successful, None otherwise.
        """
        try:
            health_check = HealthCheck(
                cluster_id=cluster_id,
                status=status,
                details=details or {},
            )
            with self.get_session() as session:
                session.add(health_check)
                session.commit()
                session.refresh(health_check)
            return health_check
        except Exception as e:
            self.logger.error(f"Error adding health check to database: {e}")
            return None
    
    def get_latest_health_check(self, cluster_id: int) -> Optional[HealthCheck]:
        """Get the latest health check for a cluster.
        
        Args:
            cluster_id: ID of the cluster.
            
        Returns:
            The latest health check if found, None otherwise.
        """
        try:
            with self.get_session() as session:
                return session.query(HealthCheck)\
                    .filter_by(cluster_id=cluster_id)\
                    .order_by(HealthCheck.timestamp.desc())\
                    .first()
        except Exception as e:
            self.logger.error(f"Error getting health check from database: {e}")
            return None
    
    # Performance metric methods
    
    def add_performance_metric(self, cluster_id: int, metric_type: str,
                              value: float, unit: str,
                              metadata: Optional[Dict[str, Any]] = None) -> Optional[PerformanceMetric]:
        """Add a performance metric entry to the database.
        
        Args:
            cluster_id: ID of the cluster.
            metric_type: Type of the metric ('cpu', 'memory', 'network', etc.).
            value: Value of the metric.
            unit: Unit of the metric.
            metadata: Optional metadata about the metric.
            
        Returns:
            The added performance metric entry if successful, None otherwise.
        """
        try:
            metric = PerformanceMetric(
                cluster_id=cluster_id,
                metric_type=metric_type,
                value=value,
                unit=unit,
                metadata=metadata or {},
            )
            with self.get_session() as session:
                session.add(metric)
                session.commit()
                session.refresh(metric)
            return metric
        except Exception as e:
            self.logger.error(f"Error adding performance metric to database: {e}")
            return None
    
    def get_performance_metrics(self, cluster_id: int, metric_type: Optional[str] = None,
                              limit: int = 100) -> List[PerformanceMetric]:
        """Get performance metrics for a cluster.
        
        Args:
            cluster_id: ID of the cluster.
            metric_type: Type of the metric to filter by, or None for all.
            limit: Maximum number of entries to return.
            
        Returns:
            List of performance metric entries.
        """
        try:
            with self.get_session() as session:
                query = session.query(PerformanceMetric).filter_by(cluster_id=cluster_id)
                if metric_type is not None:
                    query = query.filter_by(metric_type=metric_type)
                return query.order_by(PerformanceMetric.timestamp.desc()).limit(limit).all()
        except Exception as e:
            self.logger.error(f"Error getting performance metrics from database: {e}")
            return []


# Singleton instance
db_client = DatabaseClient()
