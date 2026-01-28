'use client';

import React, { useState } from 'react';
import { 
  HomeIcon,
  DocumentTextIcon,
  ChatBubbleLeftRightIcon,
  MagnifyingGlassIcon,
  ChartBarIcon,
  Cog6ToothIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline';
import { motion, AnimatePresence } from 'framer-motion';
import { useApp } from '@/hooks/useApp';
import { DetailedHealthResponse } from '@/types/api';

interface NavigationProps {
  currentTab: string;
  onTabChange: (tab: string) => void;
}

const tabs = [
  { id: 'dashboard', name: 'Dashboard', icon: HomeIcon },
  { id: 'documents', name: 'Documents', icon: DocumentTextIcon },
  { id: 'search', name: 'Search', icon: MagnifyingGlassIcon },
  { id: 'chat', name: 'Chat', icon: ChatBubbleLeftRightIcon },
  { id: 'metrics', name: 'Metrics', icon: ChartBarIcon },
];

export default function Navigation({ currentTab, onTabChange }: NavigationProps) {
  const { state } = useApp();

  const getHealthStatus = () => {
    if (!state.systemHealth) return 'unknown';
    return state.systemHealth.overall_healthy ? 'healthy' : 'unhealthy';
  };

  const healthStatus = getHealthStatus();
  const healthColor = healthStatus === 'healthy' ? 'text-green-600' : 'text-red-600';
  const HealthIcon = healthStatus === 'healthy' ? CheckCircleIcon : ExclamationTriangleIcon;

  return (
    <nav className="bg-white shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          {/* Logo and brand */}
          <div className="flex items-center">
            <div className="flex-shrink-0 flex items-center">
              <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">N</span>
              </div>
              <span className="ml-3 text-xl font-bold text-gray-900">NeuraX</span>
            </div>
          </div>

          {/* Navigation tabs */}
          <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = currentTab === tab.id;
              
              return (
                <button
                  key={tab.id}
                  onClick={() => onTabChange(tab.id)}
                  className={`
                    inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors duration-200
                    ${isActive
                      ? 'border-primary-500 text-primary-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }
                  `}
                >
                  <Icon className="h-5 w-5 mr-2" />
                  {tab.name}
                </button>
              );
            })}
          </div>

          {/* Health status */}
          <div className="flex items-center">
            <div className="flex items-center space-x-2">
              <HealthIcon className={`h-5 w-5 ${healthColor}`} />
              <span className="text-sm text-gray-600">
                {state.systemHealth ? 'System Online' : 'Checking...'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile navigation */}
      <div className="sm:hidden">
        <div className="pt-2 pb-3 space-y-1">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = currentTab === tab.id;
            
            return (
              <button
                key={tab.id}
                onClick={() => onTabChange(tab.id)}
                className={`
                  w-full flex items-center px-3 py-2 text-base font-medium transition-colors duration-200
                  ${isActive
                    ? 'bg-primary-50 border-primary-500 text-primary-700 border-l-4'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                  }
                `}
              >
                <Icon className="h-5 w-5 mr-3" />
                {tab.name}
              </button>
            );
          })}
        </div>
      </div>
    </nav>
  );
}

interface DashboardMetricsProps {
  health: DetailedHealthResponse | null;
  isLoading: boolean;
}

export function DashboardMetrics({ health, isLoading }: DashboardMetricsProps) {
  const { state } = useApp();

  if (isLoading || !health) {
    return (
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="bg-white overflow-hidden shadow rounded-lg animate-pulse">
            <div className="p-5">
              <div className="flex items-center">
                <div className="flex-1">
                  <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                  <div className="h-8 bg-gray-200 rounded w-1/2"></div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  const metrics = [
    {
      name: 'System Status',
      value: health.overall_healthy ? 'Healthy' : 'Issues Detected',
      icon: health.overall_healthy ? CheckCircleIcon : ExclamationTriangleIcon,
      color: health.overall_healthy ? 'text-green-600' : 'text-red-600',
      bgColor: health.overall_healthy ? 'bg-green-100' : 'bg-red-100',
    },
    {
      name: 'Documents',
      value: state.documents.length.toString(),
      icon: DocumentTextIcon,
      color: 'text-blue-600',
      bgColor: 'bg-blue-100',
    },
    {
      name: 'Active Components',
      value: `${Object.values(health.component_status).filter(Boolean).length}/${Object.keys(health.component_status).length}`,
      icon: Cog6ToothIcon,
      color: 'text-purple-600',
      bgColor: 'bg-purple-100',
    },
    {
      name: 'Uptime',
      value: `${Math.floor(health.uptime / 3600)}h ${Math.floor((health.uptime % 3600) / 60)}m`,
      icon: ChartBarIcon,
      color: 'text-orange-600',
      bgColor: 'bg-orange-100',
    },
  ];

  return (
    <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
      {metrics.map((metric, index) => {
        const Icon = metric.icon;
        return (
          <motion.div
            key={metric.name}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="bg-white overflow-hidden shadow rounded-lg hover:shadow-md transition-shadow"
          >
            <div className="p-5">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <div className={`${metric.bgColor} p-3 rounded-lg`}>
                    <Icon className={`h-6 w-6 ${metric.color}`} />
                  </div>
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">
                      {metric.name}
                    </dt>
                    <dd className={`text-lg font-semibold ${metric.color}`}>
                      {metric.value}
                    </dd>
                  </dl>
                </div>
              </div>
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}

interface SystemHealthProps {
  health: DetailedHealthResponse | null;
}

export function SystemHealth({ health }: SystemHealthProps) {
  if (!health) {
    return (
      <div className="bg-white shadow rounded-lg p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">System Health</h3>
        <div className="text-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto"></div>
          <p className="mt-2 text-sm text-gray-600">Loading system health...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white shadow rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium text-gray-900">System Health</h3>
        <div className={`flex items-center space-x-2 ${health.overall_healthy ? 'text-green-600' : 'text-red-600'}`}>
          {health.overall_healthy ? (
            <CheckCircleIcon className="h-5 w-5" />
          ) : (
            <ExclamationTriangleIcon className="h-5 w-5" />
          )}
          <span className="text-sm font-medium">
            {health.overall_healthy ? 'All Systems Operational' : 'System Issues Detected'}
          </span>
        </div>
      </div>

      <div className="space-y-4">
        {Object.entries(health.component_status).map(([component, status]) => (
          <div key={component} className="flex items-center justify-between">
            <span className="text-sm text-gray-600 capitalize">
              {component.replace('_', ' ')}
            </span>
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${status ? 'bg-green-400' : 'bg-red-400'}`}></div>
              <span className={`text-sm font-medium ${status ? 'text-green-600' : 'text-red-600'}`}>
                {status ? 'Online' : 'Offline'}
              </span>
            </div>
          </div>
        ))}
      </div>

      {health.component_errors && health.component_errors.length > 0 && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <h4 className="text-sm font-medium text-gray-900 mb-2">Recent Errors</h4>
          <div className="space-y-2">
            {health.component_errors.slice(0, 3).map((error, index) => (
              <div key={index} className="bg-red-50 border border-red-200 rounded-md p-3">
                <p className="text-sm text-red-700">{error}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}